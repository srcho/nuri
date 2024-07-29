import streamlit as st
import sqlite3

try:
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
except Exception as e:
    st.error(f"데이터베이스 연결 오류: {e}")

def create_user(username, password):
    c.execute("INSERT INTO users VALUES (?, ?)", (username, password))
    conn.commit()

def check_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone() is not None

def login_page():
    st.title("로그인")
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if st.session_state.logged_in:
        st.write(f"환영합니다, {st.session_state.username}님!")
        if st.button("로그아웃"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    else:
        username = st.text_input("사용자명")
        password = st.text_input("비밀번호", type="password")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("로그인"):
                if check_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("로그인 성공!")
                    st.rerun()
                else:
                    st.error("잘못된 사용자명 또는 비밀번호입니다.")
        
        with col2:
            if st.button("회원가입"):
                if username and password:
                    create_user(username, password)
                    st.success("회원가입이 완료되었습니다. 이제 로그인할 수 있습니다.")
                else:
                    st.error("사용자명과 비밀번호를 모두 입력해주세요.")

if __name__ == "__main__":
    login_page()