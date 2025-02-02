def add(n1,n2):
    return(n1 + n2)
def subtract(n1,n2):
    return(n1 -n1)
def multiply(n1, n2):
    return(n1 * n2)
def division(n1 , n2):
    return(n1 / n2)

operations = {
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": division,
}
def calculator():
    should_accumulate = True
    num1 = float(input("enter the number:"))
    while should_accumulate:
        for symbol in operations:
            print(symbol)
        operational_symbol = input("enter the symbol:")
        num2 = float(input("enter the number:"))
        answer = operations[operational_symbol](num1, num2)
        print(answer)
        choice = input(f"type 'y' if we continue the calculation with {answer}, or type 'n' to start a new calculation")
        if choice == 'y':
            num1 = answer
        else:
            should_accumulate=False
            print("/n" * 5)
            calculator()

calculator()


