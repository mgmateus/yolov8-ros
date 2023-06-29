


class A:
    def __init__(self) -> None:
       self.name = 'A'

class B:
    def __init__(self) -> None:
       self.name = 'B'

class C:
    def __init__(self) -> None:
       self.pname = 'C'

class D:
    def __init__(self) -> None:
       self.name = 'D'

class E(A, B, C, D):
    def __init__(self) -> None:
       A.__init__(self)
       B.__init__(self)
       C.__init__(self)
       D.__init__(self)

       self.name = 'E'

t = E()
print(t.pname)