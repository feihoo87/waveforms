from waveforms.security import verifyPassword, encryptPassword, InvalidKey


def verify(password, encrypted_password):
    try:
        verifyPassword(password, encrypted_password)
        return True
    except InvalidKey:
        return False


def test_verify():
    encrypted_password = encryptPassword("password")
    assert encrypted_password != "password"
    assert verify("password", encrypted_password)
    assert not verify("wrong password", encrypted_password)
