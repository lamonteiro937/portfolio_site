from django.db import models

class Project(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    notebook_url = models.URLField()

    def __str__(self):
        return self.title
