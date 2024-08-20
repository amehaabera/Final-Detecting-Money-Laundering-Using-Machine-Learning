mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"amehaabera@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]
headless = true
enableCORS = false
port = 8501
