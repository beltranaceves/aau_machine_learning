#----------------------EX1------------------------#

# STEP ONE - Gradient definiton

# starting function f(x,y) = ln(x^2+2y^2-2xy-2y-2)
def f_parx(x, y):
    num = 2*x-2*y
    den = pow(x,2)+2*pow(y,2)-2*x*y-2*y+2
    return num/den

def f_pary(x, y):
    num = 4*y-2*x-2
    den = pow(x,2)+2*pow(y,2)-2*x*y-2*y+2
    return num/den

x = -5
y = -5


v_x = 0
v_y = 0
x_m = x
y_m = y

ln = 0.5
alpha = 0.9

for i in range(7):
    x_n = x - ln * f_parx(x,y)
    y_n = y - ln * f_pary(x,y)

    v_x = alpha*v_x - ln * f_parx(x_m, y_m)
    v_y = alpha*v_y - ln * f_pary(x_m, y_m)

    x_nm = x_m + v_x
    y_nm = y_m + v_y

    x,y = x_n,y_n
    x_m,y_m = x_nm,y_nm

    print(f"iteration               {i} x1: {x} x2:{y}")
    print(f"iteration with momentum {i} x1: {x_m} x2:{y_m}")
    print("")

#----------------------EX2------------------------#
import torch

def cross_entropy_loss(t, y_pred):
    """
    Cross-entropy loss:
        L = - t * ln(y_pred) - (1 - t) * ln(1 - y_pred)
    """
    # Numerical stability: avoid log(0)
    eps = 1e-12  
    y_pred = torch.clamp(y_pred, eps, 1 - eps)

    return -t * torch.log(y_pred) - (1 - t) * torch.log(1 - y_pred)

sigmoid = torch.nn.Sigmoid()
relu = torch.nn.ReLU()

x1 = torch.tensor(1.0, requires_grad=True)
w13 = torch.tensor(2.0, requires_grad=True)

x2 = torch.tensor(3.0, requires_grad=True)
w23 = torch.tensor(0.5, requires_grad=True)

who = torch.tensor(0.5, requires_grad=True)

y1 = x1*w13
y1.register_hook(lambda grad: print("Grad y1 = {}".format(grad)))

y2 = x2*w23
y2.register_hook(lambda grad: print("Grad y2 = {}".format(grad)))

y3 = y1+y2
y3.register_hook(lambda grad: print("Grad y3 = {}".format(grad)))

y4 = relu(y3)
y4.register_hook(lambda grad: print("Grad y4 = {}".format(grad)))

y5 = y4 * who
y5.register_hook(lambda grad: print("Grad y5 = {}".format(grad)))

y6 = sigmoid(y5)
y6.register_hook(lambda grad: print("Grad y6 = {}".format(grad)))

e = cross_entropy_loss(1, y6)

e.backward()

print("Grad x1 = {}".format(x1.grad))
print("Grad x2 = {}".format(x2.grad))
print("Grad w13 = {}".format(w13.grad))
print("Grad w23 = {}".format(w23.grad))
print("Grad who = {}".format(who.grad))

print("Done")
