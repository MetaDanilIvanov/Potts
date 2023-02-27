glo = globals()
listB=[]
for i in range(1,11):
    glo["v%s" % i] = i * 10
    listB.append("v%s" % i)

def print1to10():
    print("Printing v1 to v10:")
    for i in range(1,11):
        print("v%s = " % i, end="")
        print(glo["v%s" % i])

print1to10()

listA=[]
for i in range(1,11):
    listA.append(i)

listA=tuple(listA)
print(listA, '"Tuple to unpack"')

listB = str(str(listB).strip("[]").replace("'", "") + " = listA")

print(listB)

exec(listB)

print1to10()