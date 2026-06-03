def show_auth():
    st.title("DocChat — Sign in")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username", key="li_u")
        p = st.text_input("Password", type="password", key="li_p")
        if st.button("Login"):
            if verify_user(u, p):
                st.session_state["user"] = u
                st.rerun()
            else:
                st.error("Wrong credentials")

    with tab2:
        u = st.text_input("Choose username", key="reg_u")
        p = st.text_input("Choose password", type="password", key="reg_p")
        if st.button("Register"):
            if register_user(u, p):
                st.success("Account created! Please login.")
            else:
                st.error("Username already taken")