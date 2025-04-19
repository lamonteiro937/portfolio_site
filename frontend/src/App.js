import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [projects, setProjects] = useState([]);

  useEffect(() => {
    fetch('/api/projects/')
      .then(response => response.json())
      .then(data => setProjects(data));
  }, []);

  return (
    <div className="App">
      <h1>Portfolio</h1>
      {projects.map(project => (
        <div key={project.id}>
          <h2>{project.title}</h2>
          <p>{project.description}</p>
          <p>Notebook Path: {project.notebook_path}</p>
        </div>
      ))}
    </div>
  );
}

export default App;
