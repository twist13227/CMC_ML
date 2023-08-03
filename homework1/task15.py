from typing import List


def hello(name: str = None) -> str:
    if name:
        return f'Hello, {name}!'
    else:
        return 'Hello!'

def int_to_roman(num: int) -> str:
    #Создаем массивы всех возможных вариантов каждой позиции
    ones = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']
    decades = ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC']
    hundreds = ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM']
    thousands = ['', 'M', 'MM', 'MMM']
    roman_num = thousands[num // 1000] + hundreds[num % 1000 // 100] + decades[num % 100 // 10] + ones[num % 10]
    return roman_num

def longest_common_prefix(strs_input: List[str]) -> str:
    str = ''
    if len(strs_input) != 0:
        for i in strs_input[0].strip():
            for j in strs_input:
                cur = str + i
                if j.strip().find(cur) == -1:
                    return str
            str = cur
        return str
    return str


def primes() -> int:
    cur_num = 2
    prime_flag = True
    while True:
        if prime_flag:
            yield cur_num
        prime_flag = True
        cur_num += 1
        for i in range(2, cur_num // 2 + 1):
            if cur_num % i == 0:
                prime_flag = False
                break


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, sum_spent):
        if sum_spent <= self.total_sum:
            print(f"You spent {sum_spent} dollars.")
            self.total_sum -= sum_spent
        else:
            print(f"Can't spend {sum_spent} dollars.")
            raise ValueError

    def __repr__(self):
        return "To learn the balance call balance."

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")

    def __add__(self, foreign):
        return BankCard(self.total_sum + foreign.total_sum, max(self.balance_limit, foreign.balance_limit))

    @property
    def balance(self):
        if self.balance_limit != None:
            if self.balance_limit <= 0:
                print("Balance check limits exceeded.")
                raise ValueError
            else:
                self.balance_limit -= 1
        return self.total_sum




