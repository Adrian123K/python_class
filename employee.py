class  Employees:
    raise_amount = 1.1 
    def __init__(self, first, last, pay): 
        self.first = first              
        self.last  = last
        self.pay   = pay
        self.email = first.lower() + '.' + last.lower() + '@gmail.com'
        self.raise_amount = 1.1  

    def  full_name(self): 
        return  '{} {}'.format(self.first, self.last)

    def  apply_raise(self):
        self.pay = int( self.pay * Employees.raise_amount)