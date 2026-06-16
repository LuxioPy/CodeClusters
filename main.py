import numpy as np
import pandas as pd
import random 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

term_descriptions = {
    "algorithm": "a step-by-step set of instructions or rules defined to solve a specific problem",
    "data structure": "a specialized format for organizing, processing, retrieving, and storing data",
    "machine learning": "a subset of AI that enables systems to learn and improve from experience without being explicitly programmed",
    "artificial intelligence": "the simulation of human intelligence processes by machines, especially computer systems",
    "computer vision": "a field of AI that enables computers and systems to derive meaningful information from digital images or videos",
    "natural language processing": "a branch of AI that helps computers understand, interpret, and manipulate human language",
    "deep learning": "a subset of machine learning based on artificial neural networks with multiple layers",
    "neural network": "a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the human brain",
    "reinforcement learning": "an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward",
    "supervised learning": "a type of machine learning where the model is trained on labeled data",
    "unsupervised learning": "a type of machine learning where the model looks for undetected patterns in a dataset with no pre-existing labels",
    "support vector machine": "a supervised machine learning algorithm used for both classification and regression challenges",
    "decision tree": "a non-parametric supervised learning method used for classification and regression tasks, structured like a flowchart",
    "stack": "data structure that follows last in first out order",
    "queue": "data structure that follows first in first out order",
    "linked list": "linear data structure made of nodes pointing to the next node",
    "hash table": "data structure that maps keys to values using a hash function",
    "binary tree": "a tree data structure in which each node has at most two children, referred to as the left child and the right child",
    "graph": "a non-linear data structure consisting of nodes (vertices) connected by edges",
    "sorting algorithm": "an algorithm that puts elements of a list in a certain order (e.g., numerical or alphabetical)",
    "search algorithm": "an algorithm designed to retrieve information stored within some data structure",
    "dijkstra's algorithm": "an algorithm for finding the shortest paths between nodes in a graph",
    "dynamic programming": "an algorithmic technique for solving an optimization problem by breaking it down into simpler subproblems and utilizing memoization",
    "greedy algorithm": "an algorithmic strategy that makes the locally optimal choice at each stage with the hope of finding a global optimum",
    "divide and conquer": "an algorithmic paradigm that breaks a problem down into two or more sub-problems of the same or related type until these become simple enough to be solved directly",
    "backtracking": "an algorithmic technique for solving problems recursively by trying to build a solution incrementally, removing those choices that fail to satisfy the constraints",
    "recursion": "a programming technique where a function calls itself directly or indirectly to solve a problem",
    "big O notation": "a mathematical notation that describes the limiting behavior of a function when the argument tends towards a particular value or infinity, used to classify algorithms according to their running time or space requirements",
    "time complexity": "the amount of time taken by an algorithm to run as a function of the length of the input",
    "space complexity": "the amount of memory space required by an algorithm to run to completion as a function of the length of the input",
    "parallel computing": "a type of computation in which many calculations or the execution of processes are carried out simultaneously",
    "distributed computing": "a field of computer science that studies distributed systems, where components located on networked computers communicate and coordinate their actions by passing messages",
    "cloud computing": "the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user",
    "edge computing": "a distributed computing paradigm that brings computation and data storage closer to the sources of data",
    "quantum computing": "a multidisciplinary field comprising aspects of computer science, physics, and mathematics that utilizes quantum mechanics to solve complex problems faster than on classical computers",
    "blockchain": "a decentralized, distributed ledger that records the provenance of a digital asset across a peer-to-peer network",
    "cryptography": "the practice and study of techniques for secure communication in the presence of adversarial third parties",
    "cybersecurity": "the practice of protecting systems, networks, and programs from digital attacks",
    "data mining": "the process of uncovering patterns and other valuable information from large data sets",
    "data science": "the field of study that combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data",
    "data analysis": "the process of systematically applying statistical and/or logical techniques to describe and illustrate, condense and recap, and evaluate data",
    "data visualization": "the graphic representation of data and information using visual elements like charts, graphs, and maps",
    "database": "an organized collection of structured information, or data, typically stored electronically in a computer system",
    "SQL": "Structured Query Language, a standard language for managing and manipulating relational databases",
    "NoSQL": "a class of database management systems that do not use the traditional relational model, designed for distributed data stores",
    "relational database": "a type of database that stores and provides access to data points that are related to one another based on the relational model",
    "non-relational database": "a database that does not use the tabular schema of rows and columns found in most traditional database systems",
    "object-oriented programming": "a programming paradigm based on the concept of 'objects', which can contain data and code",
    "functional programming": "a programming paradigm where programs are constructed by applying and composing functions, avoiding changing-state and mutable data",
    "procedural programming": "a programming paradigm derived from structured programming, based upon the concept of the procedure call",
    "programming language": "a system of notation for writing computer programs, which are instructions for a computer to execute",
    "compiler": "a special program that translates a programmer's source code into machine language, object code, or another programming language",
    "interpreter": "a computer program that directly executes instructions written in a programming or scripting language, without requiring them previously to have been compiled into a machine language program",
    "debugger": "a computer program used to test and debug other programs",
    "version control": "the practice of tracking and managing changes to software code",
    "git": "a distributed version control system designed to handle everything from small to very large projects with speed and efficiency",
    "software engineering": "the systematic application of engineering principles to the development of software",
    "agile development": "an iterative approach to software development and project management that helps teams deliver value to their customers faster",
    "devops": "a set of practices that combines software development and IT operations to shorten the systems development life cycle and provide continuous delivery",
    "continuous integration": "the practice of automating the integration of code changes from multiple contributors into a single software project",
    "continuous deployment": "a software engineering approach in which software functionalities are delivered frequently through automated deployments",
    "microservices": "an architectural style that structures an application as a collection of services that are highly maintainable, testable, and loosely coupled",
    "containerization": "a form of operating system virtualization where applications are run in isolated user spaces called containers",
    "docker": "an open-source platform that automates the deployment, scaling, and management of applications inside lightweight containers",
    "kubernetes": "an open-source container orchestration system for automating software deployment, scaling, and management",
    "virtualization": "the act of creating a virtual version of something, including virtual computer hardware platforms, storage devices, and computer network resources",
    "operating system": "the program that, after being initially loaded into the computer by a boot program, manages all of the other application programs in a computer",
    "linux": "an open-source, Unix-like operating system kernel that serves as the foundation for many operating systems",
    "windows": "a group of several proprietary graphical operating system families developed and marketed by Microsoft",
    "macOS": "the proprietary graphical operating system developed and marketed by Apple for its Mac family of computers",
    "networking": "the field of computer science that deals with the communication between autonomous computing devices",
    "protocol": "a set of rules governing the exchange or transmission of data between devices",
    "React": "JavaScript library for building user interfaces",
    "API": "Application Programming Interface, a set of protocols and tools that allows different software applications to communicate with each other",
    "RESTful API": "an architectural style for an API that uses HTTP requests to access and manipulate data using a stateless protocol",
    "GraphQL": "query language for APIs that lets clients request specific data",
    "Angular": "TypeScript framework for building web applications",
    "Vue.js": "JavaScript framework for building reactive user interfaces",
    "JavaScript": "a high-level, interpreted programming language that conforms to the ECMAScript specification, primarily used to create interactive web pages",
    "TCP/IP": "Transmission Control Protocol/Internet Protocol, a conceptual model and set of communications protocols used on the Internet and similar computer networks"
}

model = SentenceTransformer("all-MiniLM-L6-v2")

terms = list(term_descriptions.keys())
texts = list(term_descriptions.values())
embeddings = model.encode(texts)


similarity_matrix = cosine_similarity(embeddings)

df = pd.DataFrame(
    similarity_matrix,
    index=terms,
    columns=terms
)

print(df.round(2))

kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
labels = kmeans.fit_predict(embeddings)

groups = {}

for term, label in zip(terms, labels):
    groups.setdefault(label, []).append(term)

for group, words in groups.items():
    print(f"Group {group}: {words}")

# Define categories and their associated terms for a New York Times insipired Connections game
categories = {
    "Data Structures": ["stack", "queue", "linked list", "hash table"],
    "ML Types": ["supervised learning", "unsupervised learning", "reinforcement learning", "deep learning"],
    "Web Tools": ["React", "Angular", "Vue.js", "GraphQL"],
    "Databases": ["SQL", "NoSQL", "relational database", "non-relational database"]
}

puzzle_words = []

for category, words in categories.items():
    puzzle_words.extend(words)

random.shuffle(puzzle_words)

print(puzzle_words)


def check_category(category, selected_words): 
    guess_set = set(selected_words)
    for cat, words in categories.items():
        if guess_set == set(words):
            return True, cat
    return False, None

guess = ["stack", "queue", "linked list", "hash table"]

correct, category = check_category("Data Structures", guess)

if correct:
    print(f"Correct! Category: {category}")
else:
    print("Not quite.")


# 3d model of the embeddings using PCA for dimensionality reduction and Plotly for visualization
pca = PCA(n_components=3)
coords = pca.fit_transform(embeddings)

term_to_category = {}

for category, words in categories.items():
    for word in words:
        term_to_category[word] = category

plot_df = pd.DataFrame({
    "term": terms,
    "x": coords[:, 0],
    "y": coords[:, 1],
    "z": coords[:, 2],
    "category": [term_to_category.get(term, "Other") for term in terms]
})

fig = px.scatter_3d(
    plot_df,
    x="x",
    y="y",
    z="z",
    text="term",
    color="category",
    title="CS Connections Embedding Map"
)

fig.show()