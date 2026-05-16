def create_bio(first, last, role):
    """
    This function takes three string variables,
    concatenates them with a space, and returns a bio.
    """
    full_name = f"{first} {last}"
    bio = f"NAME: {full_name}\nROLE: {role.title()}"
    return bio

# --- Main Program ---
print("--- Welcome to the Bio Creator ---")

# Getting user input
fname = input("Enter your first name: ")
sname = input("Enter your last name: ")
job = input("Enter your desired job title: ")

# Calling the function and storing the returned value
my_bio = create_bio(fname, sname, job)

# Printing the result
print("\nYour Generated Bio:")
print("-------------------")
print(my_bio)