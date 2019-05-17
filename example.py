from qbitcontrol.templates import LZ

model = LZ(1, 1)
#print(model.lz_value)

model.state = (1, 0)
model.propagate(-10, 10, 1000)
