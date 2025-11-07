import csv
import random

def typees(c):
    if c == 0:
        return("mult")
    if c == 1:
        return("add")
    if c == 2:
        return("sub")
    if c == 3:
        return("div")

class cnf:
    bufferzone= [10_000, 25_000, 75_000, 125_000]
    bufferindex = len(bufferzone)
    max_valueforbuffer= [100,1000,10000,100000]
    calcindex = 4
    types = typees(iter([1,2,3,4]))
    mutfactor = 10

    # Parameters as the Model Grows larger



def generate_addition_problems_csv(filename="addition_problems.csv",  bufferindex = cnf.bufferindex, max_valueforbuffer= cnf.max_valueforbuffer, bufferzone= cnf.bufferzone, calcindex = cnf.calcindex, mutfactor = cnf.mutfactor):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['op','a', 'b', 'c'])
        for g in range (calcindex):
            for z in range (bufferindex):
                for i in range(bufferzone[z]):
                    a = random.randint(1, max_valueforbuffer[z] - 1)
                    b = random.randint(1, max_valueforbuffer[z] - 1)
                    if g == 0:
                        c = a * b
                    if g == 1:
                        c = a + b
                    if g == 2:
                        c = a - b
                    if g == 3:
                        
                        b = random.randint(1, max_valueforbuffer[z] // (mutfactor if (max_valueforbuffer[z]) > mutfactor else 0))
                        c = random.randint(1, max_valueforbuffer[z] // b)

                        a = b * c
                    writer.writerow([typees(g), a ,b, c])

        # Algorithm for the Number Gen..
                
            
        
                
        
    print("done")

generate_addition_problems_csv()

if True:
    input_csv = "addition_problems.csv"  # Input file from your generator
    output_txt = "problems.txt"          # Output text file

    with open(input_csv, "r", newline="") as fin, open(output_txt, "w") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            op = row["op"]
            a = row["a"]
            b = row["b"]
            c = row["c"]

            # Use symbols for readability
            symbol_map = {
                "add": "+",
                "sub": "-",
                "mult": "×",
                "div": "÷"
            }

            symbol = symbol_map.get(op, "?")
            formatted = f"{a}{symbol}{b}={c}"
            fout.write(formatted + "\n")

    print("All done! File written to:", output_txt)
