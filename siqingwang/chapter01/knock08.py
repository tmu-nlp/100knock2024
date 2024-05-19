# 08.cipher text
def cipher(message):
    ciphered_message = ""
    for char in message:
        if char.islower() and char.isalpha():
            ciphered_char = chr(219 - ord(char))
        else:
            ciphered_char = char
        ciphered_message += ciphered_char
    return ciphered_message

# Example usage
original_message = "My name is Cecilia."
ciphered_message = cipher(original_message)
deciphered_message = cipher(ciphered_message)

print("Original message:", original_message)
print("Ciphered message:", ciphered_message)
print("Deciphered message:", deciphered_message)