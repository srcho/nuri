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

def search_page():
    st.title("더 똑똑하게 찾는 학술 정보")
    
    # 검색 입력 필드와 버튼을 나란히 배치
    col1, col2 = st.columns([4, 1])  # 4:1 비율로 열 생성
    
    with col1:
        search_query = st.text_input("어떤 지식을 알고 싶으신가요?", key="search_input", label_visibility="collapsed")
    
    with col2:
        search_button = st.button("검색", key="search_button", use_container_width=True)
    
    # 키워드 태그
    keywords = ["소셜미디어", "코로나19", "AI", "OTT", "인공지능", "유튜브"]
    st.write(" ".join([f"#{keyword}" for keyword in keywords]))
    
    # 안내 메시지
    st.write("프로토타입 테스트로, 신문방송학 관련 질문만 가능합니다.")
    st.write("주요 연구 키워드를 참고하여 질문해 보세요.")

    if search_button:
        # 여기에 검색 로직을 구현합니다
        st.write(f"'{search_query}'에 대한 검색 결과:")
        # 검색 결과를 표시하는 로직을 추가합니다

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        search_page()
    else:
        login_page()  # 이전에 구현한 로그인 페이지 함수

if __name__ == "__main__":
    main()