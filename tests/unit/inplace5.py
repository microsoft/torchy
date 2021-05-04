import testdriver

x = torch.tensor(((3.,2.), (4.,5.)))
y = torch.tensor(((5.,6.), (7.,8.)))

# bump reference counter
x2 = x
w = x.add(y)
x2.add_(y)

w.add_(x)

print(x)
print(x2)
print(w)
