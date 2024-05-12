def g_text(x, y, z):
    return "{}時の{}は{}".format(x, y, z)

x = 12
y = "気温"
z = 22.4

result = g_text(x, y, z)
print(result)
