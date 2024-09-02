DB_HOST = "127.0.0.1"
DB_USER = "root" # 실무에서는 계정을 추가해서 사용! (보안상 root 계정 사용 안함)
DB_PASSWORD = "1234"
DB_NAME = "flyai"

def DatabaseConfig():
    global DB_HOST, DB_USER, DB_PASSWORD, DB_NAME