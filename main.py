import csv
from faker import Faker

fake = Faker()

# Generate data
def generate_data():
    with open('faculty_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Experience'])
        for _ in range(100000):
            writer.writerow([fake.name(), fake.random_int(min=1, max=20)])

# Load data into a list
def load_data():
    with open('faculty_data.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        return list(reader)

# Operations
def add_faculty(faculty_list, name, experience):
    faculty_list.append([name, str(experience)])

def search_faculty(faculty_list, name):
    return next((f for f in faculty_list if f[0] == name), "Not found")

def delete_faculty(faculty_list, name):
    return [f for f in faculty_list if f[0] != name]

def check_experience(faculty_list, experience):
    return [f[0] for f in faculty_list if int(f[1]) > experience]

# Menu system
def menu():
    faculty_list = load_data()
    while True:
        choice = input("1: Add, 2: Search, 3: Delete, 4: Check Exp >10, 5: Exit: ")
        if choice == '1':
            name = input("Enter name: ")
            experience = int(input("Enter experience: "))
            add_faculty(faculty_list, name, experience)
        elif choice == '2':
            name = input("Enter name to search: ")
            print(search_faculty(faculty_list, name))
        elif choice == '3':
            name = input("Enter name to delete: ")
            faculty_list = delete_faculty(faculty_list, name)
        elif choice == '4':
            experience = 10
            print(f"{', '.join(check_experience(faculty_list, experience))} can participate in BOS.")
        elif choice == '5':
            break

if __name__ == "__main__":
    generate_data()  # Uncomment this line if you need to regenerate the data
    menu()

















import csv
from faker import Faker
from datetime import datetime

fake = Faker()

# Generate data
def generate_student_data():
    with open('student_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'USN', 'CGPA', 'Address', 'Blood Group', 'Branch', 'UG/PG', 'DOB', 'Year'])
        for i in range(100):
            name = fake.name()
            usn = f"USN{i:05d}"
            cgpa = round(fake.random_number(digits=2) * 0.1, 2)
            address = fake.address()
            blood_group = fake.random_element(elements=('A+', 'B+', 'O+', 'AB+'))
            branch = fake.random_element(elements=('Computer Science', 'Electronics', 'Mechanical'))
            ug_pg = fake.random_element(elements=('UG', 'PG'))
            dob = fake.date_of_birth(minimum_age=18, maximum_age=30).strftime('%Y-%m-%d')
            year = fake.random_int(min=1, max=4)
            writer.writerow([name, usn, cgpa, address, blood_group, branch, ug_pg, dob, year])

# Load data into a tuple
def load_data():
    with open('student_data.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        return tuple(reader)

# Search and filter functions
def search_students_by_branch(students, branch_name):
    return [s for s in students if s[5] == branch_name and float(s[2]) > 9 and s[6] in ['UG', 'PG']]

# Menu system
def menu(students):
    while True:
        print("\nMenu:")
        print("1: Search Students by Branch for Placement")
        print("2: Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            branch_name = input("Enter branch name: ")
            eligible_students = search_students_by_branch(students, branch_name)
            print(f"\nEligible students in {branch_name} for placement:")
            for student in eligible_students:
                print(f"{student[0]}, {student[1]}, {student[2]}, {student[3]}, {student[4]}, {student[5]}, {student[6]}, {student[7]}, {student[8]}")
        elif choice == '2':
            print("Exiting...")
            break

if __name__ == "__main__":
    generate_student_data()  # Uncomment this line if data needs to be regenerated
    student_tuple = load_data()
    menu(student_tuple)















from faker import Faker
import random
import csv

def generate_weather():
	fake = Faker()
	
	with open('weather.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['place','temparature', 'humidity','weather'])
		for _ in range(100000):
			writer.writerow([fake.city(), round(random.uniform(-20, 40)),round(random.uniform(0, 100)),random.choice(['Sunny', 'Cloudy', 'Rainy', 'Snowy'])])

def load_data():
	with open('weather.csv', 'r') as file:
		reader=csv.reader(file)
		next(reader)
		return list(reader)


def weather_data(weather_list,city):
	
	for weather in weather_list:
		if weather[0] == city:
			print(f" \nCity : {weather[0]} \n Temperature :{weather[1]} C \n Humidity : {weather[2]} \n Weather : {weather[3]}")
			break

		


def menu():
	weather_list=load_data()

	while True:
		city = input("Enter city name : ")
		weather_data(weather_list,city)



if __name__ == "__main__":
	generate_weather()
	menu()

























import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
delayt = .1 
value = 0 # this variable will be used to store the ldr value
ldr = 7 #ldr is connected with pin number 7
led = 11
led_2 = 13#led is connected with pin number 11
GPIO.setup(led, GPIO.OUT)# as led is an output device so that’s why we set it to output.
GPIO.setup(led_2, GPIO.OUT)
GPIO.output(led, False) # keep led off by default
GPIO.output(led_2, False)
def rc_time (ldr):
    count = 0

    #Output on the pin for
    GPIO.setup(ldr, GPIO.OUT)
    GPIO.output(ldr, False)
    time.sleep(delayt)

    #Change the pin back to input
    GPIO.setup(ldr, GPIO.IN)

    #Count until the pin goes high
    while (GPIO.input(ldr) == 0):
        count += 1

    return count


#Catch when script is interrupted, cleanup correctly
try:
    # Main loop
    while True:
        print("Ldr Value:")
        value = rc_time(ldr)
        print(value)
        if ( value <= 10000 ):
                print("Lights are ON")
                GPIO.output(led_2, False)
                GPIO.output(led, True)
        else:
                print("Lights are OFF")
                GPIO.output(led, False)
                GPIO.output(led_2, True)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()



























import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def generate_faculty_data(num_rows=100000):
    data = {
        'Experience': np.arange(1, num_rows + 1),
        'Designation': ['Professor' if x > 20 else 'Associate Professor' if x > 15 else 'Assistant Professor' if x > 5 else 'Lecturer' for x in range(1, num_rows + 1)],
        'Salary': [50000 + x * 1000 for x in range(1, num_rows + 1)],
        'Publications': [2 * x for x in range(1, num_rows + 1)],
        'Book Chapters': [x // 10 for x in range(1, num_rows + 1)],
        'Consultancy Work': [x * 100 for x in range(1, num_rows + 1)],
        'Funds Received': [x * 500 for x in range(1, num_rows + 1)],
        'Professional Membership': [x % 10 == 0 for x in range(1, num_rows + 1)]
    }
    df = pd.DataFrame(data)
    df.to_csv('faculty_dataset.csv', index=False)

def load_data_to_list():
    df = pd.read_csv('faculty_dataset.csv')
    return df.values.tolist()

def calculate_correlations():
    df = pd.read_csv('faculty_dataset.csv')
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()

def linear_regression_and_plot():
    df = pd.read_csv('faculty_dataset.csv')
    X = df[['Experience']].values
    y = df['Publications'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, y_pred, color='red')
    plt.title('Linear Regression: Experience vs Publications')
    plt.xlabel('Experience')
    plt.ylabel('Publications')
    plt.show()

if __name__ == "__main__":
    generate_faculty_data()
    data_list = load_data_to_list()
    correlations = calculate_correlations()
    print(correlations)
    linear_regression_and_plot()


























import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def generate_faculty_data(num_rows=100000):
    data = {
        'Experience': np.arange(1, num_rows + 1),
        'Designation': ['Professor' if x > 20 else 'Associate Professor' if x > 15 else 'Assistant Professor' if x > 5 else 'Lecturer' for x in range(1, num_rows + 1)],
        'Salary': [50000 + x * 1000 for x in range(1, num_rows + 1)],
        'Publications': [2 * x for x in range(1, num_rows + 1)],
        'Book Chapters': [x // 10 for x in range(1, num_rows + 1)],
        'Consultancy Work': [x * 100 for x in range(1, num_rows + 1)],
        'Funds Received': [x * 500 for x in range(1, num_rows + 1)],
        'Professional Membership': [x % 10 == 0 for x in range(1, num_rows + 1)]
    }
    df = pd.DataFrame(data)
    df.to_csv('faculty_dataset.csv', index=False)

def load_data_into_tuples():
    df = pd.read_csv('faculty_dataset.csv')
    return [tuple(x) for x in df.to_records(index=False)]

def calculate_correlations():
    df = pd.read_csv('faculty_dataset.csv')
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()

def perform_linear_regression_and_plot():
    df = pd.read_csv('faculty_dataset.csv')
    X = df[['Experience']].values
    y = df['Publications'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, y_pred, color='red')
    plt.title('Experience vs Publications')
    plt.xlabel('Experience')
    plt.ylabel('Predicted Publications')
    plt.show()

if __name__ == "__main__":
    generate_faculty_data()
    tuples = load_data_into_tuples()
    correlations = calculate_correlations()
    print(correlations)
    perform_linear_regression_and_plot()
































import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def generate_faculty_data(num_rows=100000):
    data = {
        'Experience': np.arange(1, num_rows + 1),
        'Designation': ['Professor' if x > 20 else 'Associate Professor' if x > 15 else 'Assistant Professor' if x > 5 else 'Lecturer' for x in range(1, num_rows + 1)],
        'Salary': [50000 + x * 1000 for x in range(1, num_rows + 1)],
        'Publications': [2 * x for x in range(1, num_rows + 1)],
        'Book Chapters': [x // 10 for x in range(1, num_rows + 1)],
        'Consultancy Work': [x * 100 for x in range(1, num_rows + 1)],
        'Funds Received': [x * 500 for x in range(1, num_rows + 1)],
        'Professional Membership': [x % 10 == 0 for x in range(1, num_rows + 1)]
    }
    df = pd.DataFrame(data)
    df.to_csv('faculty_dataset.csv', index=False)

def load_data_into_tuples():
    df = pd.read_csv('faculty_dataset.csv')
    return [tuple(x) for x in df.to_records(index=False)]

def knn_analysis_and_plot():
    df = pd.read_csv('faculty_dataset.csv')
    X = df[['Experience']].values
    y = df['Publications'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    plt.figure(figsize=(8, 4))
    plt.scatter(X_test, y_test, color='blue', label='Actual Publications')
    plt.plot(X_test, y_pred, color='red', label='Predicted by KNN')
    plt.title('KNN Analysis: Experience vs Publications')
    plt.xlabel('Experience')
    plt.ylabel('Publications')
    plt.legend()
    plt.show()

def naive_bayes_analysis_and_plot():
    df = pd.read_csv('faculty_dataset.csv')
    X = df[['Experience']].values
    y = df['Publications'].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Encoding publications for NB

    X_train, X_test, y_train_encoded, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=0)
    nb = GaussianNB()
    nb.fit(X_train, y_train_encoded)
    y_pred_encoded = nb.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)  # Decode back to original values
    
    plt.figure(figsize=(8, 4))
    plt.scatter(X_test, y_test, color='green', label='Actual Publications')
    plt.plot(X_test, y_pred, color='orange', label='Predicted by Naive Bayes')
    plt.title('Naive Bayes Analysis: Experience vs Publications')
    plt.xlabel('Experience')
    plt.ylabel('Publications')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_faculty_data()
    tuples = load_data_into_tuples()
    knn_analysis_and_plot()
    naive_bayes_analysis_and_plot()








































import pandas as pd
import numpy as np
import re

def generate_faculty_data(num_rows=100000):
    """ Generate a dataset of faculty data with linear mappings and save to CSV. """
    data = {
        'Experience': np.arange(1, num_rows + 1),
        'Designation': ['Professor' if x > 20 else 'Associate Professor' if x > 15 else 'Assistant Professor' if x > 5 else 'Lecturer' for x in range(1, num_rows + 1)],
        'Salary': [50000 + x * 1000 for x in range(1, num_rows + 1)],
        'Publications': [2 * x for x in range(1, num_rows + 1)],
        'Book Chapters': [x // 10 for x in range(1, num_rows + 1)],
        'Consultancy Work': [x * 100 for x in range(1, num_rows + 1)],
        'Funds Received': [x * 500 for x in range(1, num_rows + 1)],
        'Professional Membership': [x % 10 == 0 for x in range(1, num_rows + 1)]
    }
    df = pd.DataFrame(data)
    df.to_csv('faculty_dataset.csv', index=False)

# Lambda function to load data into a tuple structure
load_data_into_tuples = lambda: pd.read_csv('faculty_dataset.csv').apply(tuple, axis=1).tolist()

def search_faculty():
    """ Search and display faculty details based on designation and experience using regex. """
    df = pd.read_csv('faculty_dataset.csv')
    # Regex pattern to capture faculty based on designation and experience
    pattern = r'^(Professor|Associate Professor|Assistant Professor)$'
    matches = df[
        (df['Designation'].str.match(pattern)) &
        ((df['Designation'] == 'Professor') & (df['Experience'] > 20) |
         (df['Designation'] == 'Associate Professor') & (df['Experience'] > 15) |
         (df['Designation'] == 'Assistant Professor') & (df['Experience'] > 5))
    ]

    print(matches)


if __name__ == "__main__":
    generate_faculty_data()
    faculty_tuples = load_data_into_tuples()
    search_faculty()




































import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def generate_faculty_data(num_rows=100000):
    """ Generate a faculty dataset and save to CSV. """
    data = {
        'Experience': np.arange(1, num_rows + 1),
        'Designation': ['Professor' if x > 20 else 'Associate Professor' if x > 15 else 'Assistant Professor' if x > 5 else 'Lecturer' for x in range(1, num_rows + 1)],
        'Salary': [50000 + x * 1000 for x in range(1, num_rows + 1)],
        'Publications': [2 * x for x in range(1, num_rows + 1)],
        'Book Chapters': [x // 10 for x in range(1, num_rows + 1)],
        'Consultancy Work': [x * 100 for x in range(1, num_rows + 1)],
        'Funds Received': [x * 500 for x in range(1, num_rows + 1)],
        'Professional Membership': [x % 10 == 0 for x in range(1, num_rows + 1)]
    }
    df = pd.DataFrame(data)
    df.to_csv('faculty_dataset.csv', index=False)

def load_data_into_tuples():
    """ Load the faculty data from CSV and convert into a tuple data structure. """
    df = pd.read_csv('faculty_dataset.csv')
    return [tuple(row) for row in df.itertuples(index=False)]

def association_rule_mining():
    """ Perform association rule mining on Associate Professors' contributions. """
    df = pd.read_csv('faculty_dataset.csv')
    # Filter data for Associate Professors only
    associate_df = df[df['Designation'] == 'Associate Professor']
    # Binning the data to convert into categorical data
    associate_df['Publications_Range'] = pd.cut(associate_df['Publications'], bins=[0, 50, 100, 150, 200], labels=['Low', 'Medium', 'High', 'Very High'])
    associate_df['Book_Chapters_Range'] = pd.cut(associate_df['Book Chapters'], bins=[0, 5, 10, 15, 20], labels=['Few', 'Some', 'More', 'Many'])
    associate_df['Consultancy_Work_Range'] = pd.cut(associate_df['Consultancy Work'], bins=[0, 5000, 10000, 15000], labels=['Low', 'Medium', 'High'])

    # Preparing data for association rule mining
    transactions = associate_df[['Publications_Range', 'Book_Chapters_Range', 'Consultancy_Work_Range']]
    te = TransactionEncoder()
    te_ary = te.fit(transactions.values).transform(transactions.values)
    transactions_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Applying Apriori algorithm to find frequent item sets
    frequent_itemsets = apriori(transactions_encoded, min_support=0.01, use_colnames=True)
    # Generating association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

if __name__ == "__main__":
    generate_faculty_data()
    faculty_tuples = load_data_into_tuples()
    association_rule_mining()


























import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class FacultyDataset:
    def __init__(self, num_rows=100000):
        self.num_rows = num_rows
        self.df = None
        self.filepath = 'faculty_dataset.csv'

    def generate_data(self):
        """ Generate a dataset of faculty data with linear mappings and save to CSV. """
        data = {
            'Experience': np.arange(1, self.num_rows + 1),
            'Designation': ['Professor' if x > 20 else 'Associate Professor' if x > 15 else 'Assistant Professor' if x > 5 else 'Lecturer' for x in range(1, self.num_rows + 1)],
            'Salary': [50000 + x * 1000 for x in range(1, self.num_rows + 1)],
            'Publications': [2 * x for x in range(1, self.num_rows + 1)],
            'Book Chapters': [x // 10 for x in range(1, self.num_rows + 1)],
            'Consultancy Work': [x * 100 for x in range(1, self.num_rows + 1)],
            'Funds Received': [x * 500 for x in range(1, self.num_rows + 1)],
            'Professional Membership': [x % 10 == 0 for x in range(1, self.num_rows + 1)]
        }
        self.df = pd.DataFrame(data)
        self.df.to_csv(self.filepath, index=False)

    def load_data(self):
        """ Load data from CSV into a DataFrame. """
        self.df = pd.read_csv(self.filepath)

    def find_associate_professors(self):
        if self.df is None:
            self.load_data()
        asso_prof_data = self.df[self.df['Designation'] == 'Associate Professor']  # Select all Associate Professors
        return asso_prof_data[['Experience', 'Publications', 'Book Chapters']]


    def perform_association_rule_mining(self, data):
        """ Perform association rule mining on provided DataFrame. """
        te = TransactionEncoder()
        te_ary = te.fit(data).transform(data)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Usage
if __name__ == '__main__':
    dataset = FacultyDataset()
    dataset.generate_data()
    asso_prof_25 = dataset.find_associate_professors()
    dataset.perform_association_rule_mining(asso_prof_25[['Experience', 'Publications', 'Book Chapters']])























import csv
import random
from tqdm import tqdm
from faker import Faker
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Define subjects and books mapping
cse_subjects_books = {
    "Advanced Mathematics": ["Probability and Statistics", "Probability & Statistics with Reliability, Queuing and Computer Science Applications", "Linear Algebra with Applications", "Advanced Engineering Mathematics"], 
    "ADBMS": ["Fundamentals of Database Systems", "Database System Concepts", "NoSQL for Mere Mortals"], 
    "RMI": ["Engineering Research Methodology", "Research Methods for Engineers"],
    "VR": ["Virtual Reality", "Virtual and Augmented Reality (VR/AR)"], 
    "AIML": ["Artificial Intelligence - A Modern Approach", "Machine Learning"], 
    "IoT": ["Internet of Things", "Designing the Internet of Things, Wiley"]
}
grades_map = {'S': 9, 'A': 8, 'B': 7, 'C': 6, 'D': 5}

# Function to generate books data
def generate_books(filename: str, num_records: int):
    fake = Faker()
    data = [['USN', 'SEM', 'SUB_CODE', 'SUBJECT_NAME', 'BOOK_REFERRED', 'BOOK_ID', 'GRADE_SCORED']]
    grades = ["A", "B", "C", "D", "S"]
    grades_map = {'S': 9, 'A': 8, 'B': 7, 'C': 6, 'D': 5}
    for _ in tqdm(range(num_records), desc="Generating data"):
        sem = random.randint(1, 8)
        subject = random.choice(list(cse_subjects_books.keys()))
        book_ref = random.choice(cse_subjects_books[subject])
        usn = f"1MS{24-sem}CS{fake.random_int(min=100, max=999):03}"
        sub_code = f"MCS{sem}{cse_subjects_books[subject].index(book_ref)+1}"
        book_id = f"BID{cse_subjects_books[subject].index(book_ref)+1}"
        grade = grades_map[random.choice(grades)]
        data.append([usn, sem, sub_code, subject, book_ref, book_id, grade])
    with open(filename, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)

generate_books('books.csv', 1000)

# Exception handling for file operations
try:
    df = pd.read_csv('books.csv')
    df['Grade'] = df['GRADE_SCORED'].map(grades_map).fillna(0).astype(int)
    df['Subject_Code'] = df['SUB_CODE'].astype('category').cat.codes
    df['Book_ID'] = df['BOOK_ID'].astype('category').cat.codes

    df.to_csv('extracted-books.csv', index=False, columns=['SEM', 'Subject_Code', 'Book_ID', 'Grade'])

    print("Correlation Analysis:")
    print(df[['SEM', 'Subject_Code', 'Book_ID', 'Grade']].corr())

    # Prepare data for association rule mining
    te = TransactionEncoder()
    te_ary = te.fit(df[['SEM', 'Subject_Code', 'Book_ID', 'Grade']].astype(str).values).transform(df[['SEM', 'Subject_Code', 'Book_ID', 'Grade']].astype(str).values)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    print("Association Rules:")
    print(rules)
    
except Exception as e:
    print(f"An error occurred: {e}")
























import csv
import random
from faker import Faker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up visualization
sns.set(rc={'figure.figsize':(10, 6)})

# Mapping for subjects and corresponding books
cse_subjects_books = {
    "Advanced Mathematics": ["Probability and Statistics", "Probability & Statistics with Reliability, Queuing and Computer Science Applications", "Linear Algebra with Applications", "Advanced Engineering Mathematics"], 
    "ADBMS": ["Fundamentals of Database Systems", "Database System Concepts", "NoSQL for Mere Mortals"], 
    "RMI": ["Engineering Research Methodology", "Research Methods for Engineers"],
    "VR": ["Virtual Reality", "Virtual and Augmented Reality (VR/AR)"], 
    "AIML": ["Artificial Intelligence - A Modern Approach", "Machine Learning"], 
    "IoT": ["Internet of Things", "Designing the Internet of Things, Wiley"]
}

# Grades mapping for conversion
grades_map = {'S': 9, 'A': 8, 'B': 7, 'C': 6, 'D': 5}

# Generate synthetic data and save to CSV
def generate_and_save_books(filename: str, num_records: int):
    fake = Faker()
    data = [['USN', 'Semester Number', 'Subject Code', 'Subject Name', 'Book Referred', "Book ID", "Grade Scored"]]
    for _ in range(num_records):
        subject, books = random.choice(list(cse_subjects_books.items()))
        book = random.choice(books)
        data.append([
            fake.unique.random_int(min=10000, max=99999),  # Student USN
            random.randint(1, 8),  # Semester Number
            subject[:3] + str(random.randint(100, 999)),  # Subject Code
            subject,  # Subject Name
            book,  # Book Referred
            books.index(book) + 1,  # Book ID
            random.choice(list(grades_map.keys()))  # Grade Scored
        ])
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Read and process CSV data
def process_and_analyze_data(input_filename):
    try:
        df = pd.read_csv(input_filename)
        df['Grade Scored'] = df['Grade Scored'].map(grades_map)
        # Ensure that only numeric data is involved in correlation calculations
        numeric_df = df[['Semester Number', 'Book ID', 'Grade Scored']].dropna()
        numeric_df.to_csv("extracted-books.csv", index=False)
        print("Numeric data extracted to extracted-books.csv.")

        # Correlation analysis
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

        matrix = numeric_df.pivot_table(index='Book ID', values='Grade Scored', aggfunc='mean', fill_value=0)
        similarity = matrix.T.corr(method='pearson')  # Pearson's correlation for similarity
        print("Collaborative Filtering - Similarity Matrix:")
        print(similarity)

    except Exception as e:
        print("An error occurred:", e)

# Execute functions
generate_and_save_books('books.csv', 1000)
process_and_analyze_data('books.csv')





































import csv
import random
from faker import Faker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Setup and mappings
faker = Faker()
subjects = ['Mathematics', 'Physics', 'Literature', 'Computer Science']
books = {s: [f"{s} Book {i}" for i in range(1, 4)] for s in subjects}
grades_map = {'S': 9, 'A': 8, 'B': 7, 'C': 6, 'D': 5}

def generate_and_process_books(num_records):
    # Generate synthetic data
    data = [['USN', 'Semester Number', 'Subject Code', 'Subject Name', 'Book Referred', 'Book ID', 'Grade Scored']]
    for _ in range(num_records):
        subject = random.choice(subjects)
        book = random.choice(books[subject])
        data.append([
            faker.unique.random_int(min=1000, max=9999),
            random.randint(1, 8),
            f"{subject[:3].upper()}{random.randint(100, 999)}",
            subject,
            book,
            books[subject].index(book) + 1,
            random.choice(list(grades_map.keys()))
        ])

    # Save to CSV
    with open('books.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # Read and process data
    df = pd.read_csv('books.csv')
    df['Grade Scored'] = df['Grade Scored'].map(grades_map)
    numeric_df = df[['Semester Number', 'Book ID', 'Grade Scored']].dropna()
    numeric_df.to_csv('extracted-books.csv', index=False)

    # Plot correlation matrix
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

def mine_association_rules():
    # Association rule mining
    df = pd.read_csv('extracted-books.csv')
    df['Book ID'] = 'BookID_' + df['Book ID'].astype(str)
    df['Semester Number'] = 'Sem_' + df['Semester Number'].astype(str)
    df['Grade Scored'] = 'Grade_' + df['Grade Scored'].astype(str)

    transactions = df.values.tolist()
    te = TransactionEncoder()
    df_encoded = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    print("Association Rules Found:", rules)

if __name__ == "__main__":
    generate_and_process_books(100)
    mine_association_rules()




























import csv
import random
from faker import Faker
import pandas as pd
import numpy as np  # Correctly import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def setup_data(num_records):
    """Generates and processes data, then performs association rule mining."""
    faker = Faker()
    subjects = ['Mathematics', 'Physics', 'Literature', 'Computer Science']
    books = {s: [f"{s} Book {i}" for i in range(1, 4)] for s in subjects}
    grades_map = {'S': 9, 'A': 8, 'B': 7, 'C': 6, 'D': 5}

    # Generate synthetic data
    data = [['USN', 'Semester Number', 'Subject Code', 'Subject Name', 'Book Referred', 'Book ID', 'Grade Scored']]
    for _ in range(num_records):
        subject = random.choice(subjects)
        book = random.choice(books[subject])
        data.append([
            faker.unique.random_int(min=1000, max=9999),
            random.randint(1, 8),
            f"{subject[:3].upper()}{random.randint(100, 999)}",
            subject,
            book,
            books[subject].index(book) + 1,
            random.choice(list(grades_map.keys()))
        ])
    
    # Save and process the data
    with open('books.csv', 'w', newline='') as file:
        csv.writer(file).writerows(data)
    print(f"Data generated and saved to books.csv")

    df = pd.read_csv('books.csv')
    df['Grade Scored'] = df['Grade Scored'].map(grades_map)
    df.to_csv('extracted-books.csv', index=False)
    print("Processed data saved to extracted-books.csv")

    # Correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    # Association rule mining
    te = TransactionEncoder()
    transactions = df[['Semester Number', 'Book ID', 'Grade Scored']].applymap(str).values.tolist()
    df_encoded = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    print("Association Rules Found:")
    print(rules)

if __name__ == "__main__":
    setup_data(100)
