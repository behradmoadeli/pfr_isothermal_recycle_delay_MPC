a = True
b = 0
while a:
    b += 1
    print(b)
    if b > 3:
        a = False
        print('checkpoint 1')
        continue
    print('checkpoint 2')