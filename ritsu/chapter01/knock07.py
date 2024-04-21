def generate_txt(x, y, z):
    """
    x: x時の
    y: yは
    z: z
    """
    return f"{x}時の{y}は{z}"

result = generate_txt(12, "気温", 22.4)
print(result)