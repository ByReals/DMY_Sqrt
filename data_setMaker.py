import csv
import random

def generate_addition_problems_csv(filename="addition_problems.csv", num_samples=200_000, max_value=10000):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['a', 'b', 'c'])
        for _ in range(num_samples):
            a = random.randint(1, max_value - 1)
            b = random.randint(1, max_value - 1)
            c = a + b
            writer.writerow([a ,"+", b, "=", c])
    print("done")

generate_addition_problems_csv()

input_csv = "addition_problems.csv"  # Your CSV file
output_txt = "additions.txt"

with open(input_csv, "r") as fin, open(output_txt, "w") as fout:
    # Skip header
    next(fin)
    for line in fin:
        parts = line.strip().split(",")
        # parts = [a, "+", b, "=", c]
        if len(parts) == 5:
            a, plus, b, equals, c = parts
            # Make sure plus and equals are correct symbols if you want to validate
            formatted = f"{a}+{b}={c}"
            fout.write(formatted + "\n")
        else:
            print("Skipping malformed line:", line)

print("all done!")