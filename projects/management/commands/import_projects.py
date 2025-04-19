import os
import json
from django.core.management.base import BaseCommand
from projects.models import Project

class Command(BaseCommand):
    help = 'Imports projects from the portfolio_projects directory'

    def handle(self, *args, **options):
        projects_dir = 'portfolio_projects'
        for filename in os.listdir(projects_dir):
            if filename.endswith('.ipynb'):
                filepath = os.path.join(projects_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                    # Extract project title and description from the notebook
                    title = notebook_data['metadata'].get('title', filename[:-6])  # Use filename as title if not found in metadata
                    description = notebook_data['metadata'].get('description', 'No description available')

                    # Create the project
                    project = Project(title=title, description=description, notebook_path=filepath)
                    project.save()

                self.stdout.write(self.style.SUCCESS(f'Successfully imported project "{title}"'))
