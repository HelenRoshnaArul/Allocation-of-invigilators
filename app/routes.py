from flask import Blueprint, request, render_template, send_from_directory
import os
import pandas as pd
from genetic_algorithm import genetic_algorithm, load_datasets

main = Blueprint('main', __name__, static_folder='static')

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/import', methods=['POST'])
def import_data():
    invigilators_file = request.files['invigilators']
    availability_file = request.files['availability']
    exam_schedule_file = request.files['exam_schedule']

    # Reading the CSV files
    invigilators_df, availability_df, exam_schedule_df = load_datasets(invigilators_file, availability_file, exam_schedule_file)

    # Execute the genetic algorithm
    best_allocation ,fitness_history = genetic_algorithm(10, 50, exam_schedule_df, invigilators_df, availability_df, mutation_rate=0.1)

    # Convert the best allocation to DataFrame
    allocation_df = best_allocation.to_dataframe()

    # Create output directory if it does not exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    output_file = os.path.join(output_dir, 'invigilator_allocation_genetic.csv')
    allocation_df.to_csv(output_file, index=False)

    # Debugging: Check if the file is saved correctly
    if os.path.exists(output_file):
        print(f"File saved successfully at {output_file}")
    else:
        print(f"File failed to save at {output_file}")

    # Convert DataFrame to HTML table
    allocation_html = allocation_df.to_html(classes='table table-striped', header=True, index=False)

    # Render the template with the results
    return render_template('results.html', table=allocation_html)

@main.route('/download')
def download_file():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    return send_from_directory(output_dir, 'invigilator_allocation_genetic.csv', as_attachment=True)
