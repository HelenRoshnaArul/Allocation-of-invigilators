import pandas as pd
import numpy as np
import random
from collections import defaultdict

# Load datasets
def load_datasets(invigilators_path, availability_path, exam_schedule_path):
    availability_df = pd.read_csv(availability_path)
    exam_schedule_df = pd.read_csv(exam_schedule_path)
    invigilators_df = pd.read_csv(invigilators_path)

    # Convert date columns to consistent format
    availability_df['Date'] = pd.to_datetime(availability_df['Date']).dt.strftime('%Y-%m-%d')
    exam_schedule_df['Exam_Date'] = pd.to_datetime(exam_schedule_df['Exam_Date'], format='%d-%b-%y').dt.strftime('%Y-%m-%d')

    return invigilators_df, availability_df, exam_schedule_df

# Precompute special qualifications and availability
def precompute_data(invigilators_df, availability_df):
    special_qualified_invigilators = set(invigilators_df[invigilators_df['SpecialQualification'] == 1]['Invigilator_ID'])
    availability_map = availability_df.groupby(['Date', 'Time', 'Availability_Status'])['Invigilator_ID'].apply(set).to_dict()
    return special_qualified_invigilators, availability_map

# Define the Invigilator class
class Invigilator:
    def __init__(self, invigilator_id, name, lead, special):
        self.id = invigilator_id
        self.name = name
        self.lead = lead
        self.special = special

# Define the Venue class
class Venue:
    def __init__(self, date, time, name, capacity, special):
        self.date = date
        self.time = time
        self.name = name
        self.capacity = capacity
        self.special = special

class Allocation:
    def __init__(self):
        self.allocations = []

    def add_allocation(self, venue, invigilators, lead, special_skill_invigilators, normal_invigilators, backups):
        total_invigilators = len(invigilators)
        self.allocations.append({
            'Date': venue.date,
            'Time': venue.time,
            'Venue': venue.name,
            'Capacity': venue.capacity,
            'Total Invigilators': total_invigilators,
            'Lead Invigilator': lead,
            'Special_Requirement': 'yes' if venue.special else 'no',
            'Special Skill Invigilators': special_skill_invigilators,
            'Normal Invigilators': normal_invigilators,
            'Backup Invigilators': backups
        })

    def to_dataframe(self):
        return pd.DataFrame(self.allocations)

# Initialize the population with random allocations
def initialize_population(pop_size, exam_schedule, invigilators, availability, special_qualified_invigilators, availability_map):
    return [allocate_invigilators_randomly(exam_schedule, invigilators, availability, special_qualified_invigilators, availability_map) for _ in range(pop_size)]

def allocate_invigilators_randomly(exam_schedule, invigilators, availability, special_qualified_invigilators, availability_map):
    allocation = Allocation()
    invigilator_counts = defaultdict(int)
    invigilator_schedule = defaultdict(lambda: {'AM': False, 'PM': False})
    
    all_invigilators = set(invigilators['Invigilator_ID'].tolist())

    for _, exam in exam_schedule.iterrows():
        date, time, venue = exam['Exam_Date'], exam['TimeSlot'], exam['Venue_Name']
        capacity = exam['Capacity']
        special_requirement = exam['Special_Requirement']

        # Determine the number of needed invigilators based on capacity and special requirements
        invigilator_ratio = 3 if special_requirement else 30
        needed_invigilators = max(1, capacity // invigilator_ratio)
        available_invigilators = availability_map.get((date, time, 'available'), set())
        
        if special_requirement == 1:
            available_invigilators &= special_qualified_invigilators

        # Filter out invigilators who are already assigned to a conflicting session on the same day
        available_invigilators = [inv for inv in available_invigilators if not invigilator_schedule[inv][time]]
        available_invigilators = sorted(list(available_invigilators), key=lambda x: invigilator_counts[x])
        
        # Ensure we don't over-assign invigilators based on the venue's capacity
        selected_invigilators = random.sample(available_invigilators, min(len(available_invigilators), needed_invigilators))
        
        lead_invigilator = None
        special_skill_invigilators, normal_invigilators = [], []
        
        for invigilator_id in selected_invigilators:
            invigilator_info = invigilators.loc[invigilators['Invigilator_ID'] == invigilator_id].iloc[0]
            if lead_invigilator is None and invigilator_info['LeadInvigilator'] == 'Yes':
                if special_requirement == 0 or (special_requirement == 1 and invigilator_info['SpecialQualification'] == 1):
                    lead_invigilator = invigilator_info['Invigilator_Name']

        # If no lead invigilator was assigned, force assign the first invigilator as lead
        if lead_invigilator is None and selected_invigilators:
            lead_invigilator = invigilators.loc[invigilators['Invigilator_ID'] == selected_invigilators[0]].iloc[0]['Invigilator_Name']
        
        for invigilator_id in selected_invigilators:
            invigilator_info = invigilators.loc[invigilators['Invigilator_ID'] == invigilator_id].iloc[0]
            if invigilator_info['Invigilator_Name'] != lead_invigilator:
                if special_requirement == 1 and invigilator_info['SpecialQualification'] == 1:
                    special_skill_invigilators.append(invigilator_info['Invigilator_Name'])
                else:
                    normal_invigilators.append(invigilator_info['Invigilator_Name'])

        # Ensure the final number of normal invigilators is correct
        normal_invigilators = normal_invigilators[:max(0, needed_invigilators - len(special_skill_invigilators) - 1)]

        # Assign invigilators to avoid any session being completely unstaffed
        if not selected_invigilators and available_invigilators:
            selected_invigilators = available_invigilators[:needed_invigilators]  # Take whatever is available

        for invigilator_id in selected_invigilators:
            invigilator_counts[invigilator_id] += 1
            invigilator_schedule[invigilator_id][time] = True  # Mark the invigilator as busy for this time slot
        
        backup_invigilators = allocate_backup_invigilators(
            date, time, capacity, special_requirement,
            invigilators, availability_map,
            special_qualified_invigilators,
            invigilator_counts, defaultdict(int),
            selected_invigilators
        )
        
        allocation.add_allocation(Venue(date, time, venue, capacity, special_requirement), selected_invigilators, lead_invigilator, special_skill_invigilators, normal_invigilators, backup_invigilators)

    # Force assign any unassigned invigilators to ensure all invigilators are used at least once
    allocation = force_assign_unassigned_invigilators(allocation, invigilators, availability)

    return allocation


# Function to force assign unassigned invigilators
def force_assign_unassigned_invigilators(allocation, invigilators, availability):
    assigned_invigilators = set()

    # Gather all assigned invigilators from the current allocation
    for alloc in allocation.allocations:
        assigned_invigilators.add(alloc['Lead Invigilator'])
        assigned_invigilators.update(alloc['Special Skill Invigilators'])
        assigned_invigilators.update(alloc['Normal Invigilators'])
        assigned_invigilators.update(alloc['Backup Invigilators'])

    all_invigilators = set(invigilators['Invigilator_Name'])
    unassigned_invigilators = list(all_invigilators - assigned_invigilators)

    # Assign unassigned invigilators to available slots
    for invigilator in unassigned_invigilators:
        available_slots = availability[(availability['Invigilator_Name'] == invigilator)]

        if not available_slots.empty:
            for _, slot in available_slots.iterrows():
                date, time = slot['Date'], slot['Time']
                available_exams = [alloc for alloc in allocation.allocations if alloc['Date'] == date and alloc['Time'] == time]


    return allocation

def allocate_backup_invigilators(date, time, capacity, special_requirement, invigilators, availability_map, special_qualified_invigilators, invigilator_counts, backup_invigilator_counts, selected_invigilators):
    backup_needed = 3 if capacity >= 30 else 1
    available_invigilators = availability_map.get((date, time, 'available'), set()) - set(selected_invigilators)

    if special_requirement == 1:
        available_invigilators &= special_qualified_invigilators

    available_invigilators = sorted(list(available_invigilators), key=lambda x: backup_invigilator_counts[x])
    backup_invigilators = random.sample(available_invigilators, min(len(available_invigilators), backup_needed))

    invigilator_df = invigilators.set_index('Invigilator_ID')
    backup_invigilator_names = [invigilator_df.loc[invigilator_id]['Invigilator_Name'] for invigilator_id in backup_invigilators]

    for invigilator_id in backup_invigilators:
        backup_invigilator_counts[invigilator_id] += 1

    return backup_invigilator_names

def fitness(allocation, invigilators, max_assignments=3):
    score = 0
    invigilator_counts = defaultdict(int)
    all_invigilators = set(invigilators['Invigilator_ID'].tolist())
    assigned_invigilators = set()
    invigilator_schedule = defaultdict(lambda: {'AM': False, 'PM': False})

    for alloc in allocation.allocations:
        date = alloc['Date']
        time = alloc['Time']
        lead_invigilator = alloc['Lead Invigilator']
        special_skill_invigilators = alloc['Special Skill Invigilators']
        normal_invigilators = alloc['Normal Invigilators']
        backup_invigilators = alloc['Backup Invigilators']

        # Lead Invigilator Presence
        if lead_invigilator:
            score += 10

        # Special Requirements Met
        if alloc['Special_Requirement'] == 'yes' and len(special_skill_invigilators) >= 1:
            score += 10

        # Adequate Number of Invigilators
        needed_invigilators = max(1, alloc['Capacity'] // (3 if alloc['Special_Requirement'] == 'yes' else 30))
        total_invigilators = len(special_skill_invigilators) + len(normal_invigilators) + (1 if lead_invigilator else 0)
        if total_invigilators >= needed_invigilators:
            score += 10

        # Balanced Distribution of Assignments
        all_assigned_invigilators = special_skill_invigilators + normal_invigilators + [lead_invigilator] + backup_invigilators
        for invigilator in all_assigned_invigilators:
            if invigilator:
                assigned_invigilators.add(invigilator)
                invigilator_counts[invigilator] += 1
                score += 5  # Score for each assignment
                if invigilator_counts[invigilator] > max_assignments:
                    score -= 3  # Penalty for over-assignment

                # No Clashes (AM/PM)
                if time in ['AM', 'PM']:
                    if invigilator_schedule[invigilator][time]:
                        score -= 10  # Heavy penalty for double-booking
                    invigilator_schedule[invigilator][time] = True

        # Backup Invigilators
        if len(backup_invigilators) > 0:
            score += len(backup_invigilators)  # Add points for backup invigilators
            if len(backup_invigilators) > 3:
                score -= 3  # Penalty if too many backup invigilators

    # Full Coverage Penalty
    if not all_invigilators.issubset(assigned_invigilators):
        score -= 15

    return score

# Selection function
def selection(population, fitness_scores):
    total_fitness = np.sum(fitness_scores)
    probabilities = fitness_scores / total_fitness
    selected_indices = np.random.choice(len(population), size=len(population)//2, replace=False, p=probabilities)
    return [population[i] for i in selected_indices]

# Crossover function
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1.allocations) - 1)
    child1 = Allocation()
    child2 = Allocation()
    child1.allocations = parent1.allocations[:crossover_point] + parent2.allocations[crossover_point:]
    child2.allocations = parent2.allocations[:crossover_point] + parent1.allocations[crossover_point:]
    return child1, child2

def mutation(allocation, invigilators, availability_map, special_qualified_invigilators, mutation_rate=0.1):
    if random.random() < mutation_rate:
        index = random.randint(0, len(allocation.allocations) - 1)
        exam = allocation.allocations[index]
        date, time, venue = exam['Date'], exam['Time'], exam['Venue']
        capacity = exam['Capacity']
        special_requirement = 1 if exam['Special_Requirement'] == 'yes' else 0

        needed_invigilators = max(1, capacity // (3 if special_requirement == 1 else 30))

        available_invigilators = availability_map.get((date, time, 'available'), set())
        if special_requirement == 1:
            available_invigilators &= special_qualified_invigilators

        available_invigilators = list(available_invigilators)

        if len(available_invigilators) < needed_invigilators:
            additional_invigilators = invigilators.query('SpecialQualification == 1 & Invigilator_ID in @available_invigilators')['Invigilator_ID'].tolist()
            available_invigilators.extend(additional_invigilators)

        selected_invigilators = random.sample(available_invigilators, min(len(available_invigilators), needed_invigilators))

        lead_invigilator, special_skill_invigilators, normal_invigilators = None, [], []

        for invigilator_id in selected_invigilators:
            invigilator_info = invigilators.loc[invigilators['Invigilator_ID'] == invigilator_id].iloc[0]
            if lead_invigilator is None and invigilator_info['LeadInvigilator'] == 'Yes':
                lead_invigilator = invigilator_info['Invigilator_Name']
            elif special_requirement == 1:
                special_skill_invigilators.append(invigilator_info['Invigilator_Name'])
            else:
                normal_invigilators.append(invigilator_info['Invigilator_Name'])

        if lead_invigilator is None and selected_invigilators:
            lead_invigilator = invigilators.loc[invigilators['Invigilator_ID'] == selected_invigilators[0]].iloc[0]['Invigilator_Name']

        if lead_invigilator in normal_invigilators:
            normal_invigilators.remove(lead_invigilator)
        if lead_invigilator in special_skill_invigilators:
            special_skill_invigilators.remove(lead_invigilator)

        backup_invigilators = allocate_backup_invigilators(date, time, capacity, special_requirement, invigilators, availability_map, special_qualified_invigilators, defaultdict(int), defaultdict(int), selected_invigilators)

        allocation.allocations[index] = {
            'Date': date,
            'Time': time,
            'Venue': venue,
            'Capacity': capacity,
            'Total Invigilators': len(selected_invigilators),
            'Lead Invigilator': lead_invigilator,
            'Special_Requirement': 'yes' if special_requirement == 1 else 'no',
            'Special Skill Invigilators': special_skill_invigilators,
            'Normal Invigilators': normal_invigilators,
            'Backup Invigilators': backup_invigilators
        }

# Genetic algorithm function
def genetic_algorithm(pop_size, generations, exam_schedule, invigilators, availability, mutation_rate=0.1, max_assignments=3):
    # Precompute the necessary data
    special_qualified_invigilators, availability_map = precompute_data(invigilators, availability)
    
    # Initialize the population with proper arguments
    population = [allocate_invigilators_randomly(exam_schedule, invigilators, availability, special_qualified_invigilators, availability_map) for _ in range(pop_size)]

    fitness_history = []

    for gen in range(generations):
        fitness_scores = [fitness(individual, invigilators, max_assignments=max_assignments) for individual in population]
        fitness_history.append(fitness_scores)
        
        selected_population = selection(population, fitness_scores)

        next_generation = []
        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            mutation(child1, invigilators, availability_map, special_qualified_invigilators, mutation_rate)
            mutation(child2, invigilators, availability_map, special_qualified_invigilators, mutation_rate)
            next_generation.extend([child1, child2])

        next_generation = next_generation[:pop_size]

        # Preserve a portion of the best individuals (elitism)
        elite_count = pop_size // 10
        best_individuals = sorted(population, key=lambda ind: fitness(ind, invigilators, max_assignments), reverse=True)[:elite_count]
        population = next_generation[:pop_size - elite_count] + best_individuals

    best_individual = max(population, key=lambda individual: fitness(individual, invigilators, max_assignments=max_assignments))
    
    return best_individual, fitness_history

