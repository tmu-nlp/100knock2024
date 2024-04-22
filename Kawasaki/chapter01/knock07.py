def Template(x, y, z):
    temp = str(x)+ "時の" + str(y)+ "は" + str(z)
    return temp

result = Template(12, "気温", 22.4)
print(result)