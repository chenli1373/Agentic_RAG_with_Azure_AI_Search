import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
import uuid
import json
# from dotenv import load_dotenv
# load_dotenv()

class Memory:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self._create_memory_table()
        self._create_session_index_table()
        self._create_user_session_table()
        self._create_bad_case_table()
        self._create_indexed_sources_table()
    
    # 连接数据库
    def _get_connection(self):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn
    
    """
    #############################################
    记忆相关
    #############################################
    """
    # 创建表
    def _create_memory_table(self):
        conn = self._get_connection()
        conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                        (user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        user_query TEXT,
                        gpt_response TEXT,
                        model TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.close()

    # 插入对话
    def insert(self, session_id: str, user_query: str, response: str, model: str):
        conn = self._get_connection()
        conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                    (session_id, user_query, response, model))
        conn.commit()
        conn.close()
    
    # 获取前 limit 个历史对话
    def get_history(self, session_id: str, limit: int = 5) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
                    SELECT user_query, gpt_response 
                    FROM application_logs 
                    WHERE session_id = ? 
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''',
                    (session_id, limit))
        rows = cursor.fetchall()[:: -1]
        conn.close()

        messages = []
        for row in rows:
            messages.extend([
                {"role": "human", "content": row["user_query"]},
                {"role": "ai", "content": row["gpt_response"]}
            ])
        return messages
    
    # 清除所有对话
    def clear_memory(self, session_id: str):
        conn = self._get_connection()
        conn.execute('DELETE FROM application_logs WHERE session_id = ?',
            (session_id,))
        conn.commit()
        conn.close()

    """
    #############################################
    session_id 和 index_name 映射表
    #############################################
    """
    # 创建 session_id 和 index_name 的映射表
    def _create_session_index_table(self):
        conn = self._get_connection()
        conn.execute('''CREATE TABLE IF NOT EXISTS session_index_map(
                        session_id TEXT PRIMARY KEY,
                        index_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.close()
    
    # 保存 session_id 和 index_name 映射
    def save_session_index(self, session_id: str, index_name: str):
        conn = self._get_connection()
        conn.execute('''INSERT OR REPLACE INTO session_index_map (session_id, index_name)
                     VALUE (?, ?)''', (session_id, index_name))
        conn.commit()
        conn.close()

    # 获取 session_id 对应的 index_name
    def get_index_name(self, session_id: str) -> Optional[str]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
                SELECT index_name FROM session_index_map
                       WHERE session_id = ?
                ''', (session_id))
        row = cursor.fetchone()
        conn.close()

        return row["index_name"] if row else None
    
    def clear_session_index(self, session_id: str):
        conn = self._get_connection()
        conn.execute('DELETE FROM session_index_map WHERE session_id = ?',
            (session_id,))
        conn.commit()
        conn.close()
    
    """
    #############################################
    user_id 和 session_id 映射表
    #############################################
    """
    def _create_user_session_table(self):
        conn = self._get_connection()
        conn.execute("""CREATE TABLE IF NOT EXISTS user_sessions (
                        user_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL)""")
        conn.close()

    def get_or_create_session(self, user_id: str):
        conn = self._get_connection()

        cursor = conn.execute('''SELECT session_id FROM user_sessions WHERE user_id = ?''', (user_id,))
        row = cursor.fetchone()

        if row:
            session_id = row["session_id"]
        else:
            session_id = str(uuid.uuid4())
            conn.execute('''INSERT INTO user_sessions (user_id, session_id) VALUES (?, ?)''', (user_id, session_id))
            conn.commit()
        conn.close()
        return session_id
    
    def clear_user_session(self, session_id: str):
        conn = self._get_connection()
        conn.execute('DELETE FROM user_sessions WHERE session_id = ?',
            (session_id,))
        conn.commit()
        conn.close()
    
    """
    #############################################
    bad case 库
    #############################################
    """
    def _create_bad_case_table(self):
        conn = self._get_connection()
        conn.execute("""CREATE TABLE IF NOT EXISTS bad_cases (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,

                        stage TEXT,              -- query_refine / retrieval / answer / critic
                        query TEXT,

                        input TEXT,              -- 输入 state 或 prompt（JSON string）
                        output TEXT,             -- 模型输出（JSON or text）

                        score REAL,              -- critic score
                        reason TEXT,             -- critic reason / failure reason

                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.close()
    
    def save_bad_cases(self,
                       session_id: str,
                        stage: str,
                        query: str,
                        input_data,
                        output_data,
                        score: float,
                        reason: str):
        conn = self._get_connection()
        cursor = conn.cursor()

        # 统一序列化（防止 dict/list 直接写入报错）
        input_str = json.dumps(input_data, ensure_ascii=False) if isinstance(input_data, (dict, list)) else str(input_data)
        output_str = json.dumps(output_data, ensure_ascii=False) if isinstance(output_data, (dict, list)) else str(output_data)

        cursor.execute("""INSERT INTO bad_cases (
                       session_id, stage, query, input, output, score, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (session_id, stage, query, input_str, output_str, score, reason))
        conn.commit()
        conn.close()

    def get_bad_cases_for_prompt(self, session_id: str, limit: int = 3):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""SELECT stage, query, input, output, score, reason
                            FROM bad_cases
                            WHERE session_id = ?
                            ORDER BY score ASC, created_at DESC
                            LIMIT ? """, (session_id, limit))
        rows = cursor.fetchall()
        conn.close()

        formatted = []

        for r in rows:
            formatted.append(f"""
                ### BAD CASE ({r['stage']})

                Query:
                {r['query']}

                Input:
                {r['input']}

                Output:
                {r['output']}

                Score:
                {r['score']}

                Reason:
                {r['reason']}

                IMPORTANT LESSON:
                - Avoid repeating this mistake
                - Do not imitate the output format
                - Ensure answer is fully grounded in retrieved documents
                """)
        return "\n".join(formatted)
    
    def clear_bad_case(self, session_id: str):
        conn = self._get_connection()
        conn.execute('DELETE FROM bad_cases WHERE session_id = ?',
            (session_id,))
        conn.commit()
        conn.close()
    
    def delete_bad_case_table(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DROP TABLE IF EXISTS bad_cases")
            conn.commit()
            print("bad_cases 表结构已删除")
        except Exception as e:
            print(f"删除 bad_cases 表失败: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    """
    #############################################
    索引记录表，记录哪些内容已经上传
    #############################################
    """
    def _create_indexed_sources_table(self):
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT UNIQUE,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.close()

    # 判断 source 是否已经上传
    def is_source_indexed(self, source: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM indexed_sources WHERE source = ?",
                        (source,))
        
        result = cursor.fetchone()
        conn.close()

        return result is not None
    
    # 记录上传的 source
    def mark_source_indexed(self, source: str):
        conn = self._get_connection()
        conn.execute("INSERT OR IGNORE INTO indexed_sources (source) VALUES (?)",
                    (source,))
        
        conn.commit()
        conn.close()

    def clear_source_indexed(self, session_id: str):
        conn = self._get_connection()
        conn.execute('DELETE FROM indexed_sources WHERE session_id = ?',
            (session_id,))
        conn.commit()
        conn.close()


# if __name__ == "__main__":
#     memory = Memory(db_name=os.getenv("DB_NAME"))
#     print("初始化成功")
    