// src/main.ts with SimulationPersistence integration
import { Simulation } from './core/sim';
import { Renderer } from './visualization/renderer';
import { DataRecorder } from './analytics/data-recorder';
import { SimulationPersistence } from './persistence/simulation-saver';
import config from './cfg';

document.addEventListener('DOMContentLoaded', () => {
  console.log('Evolution Simulation initializing...');

  // Get simulation container
  const simulationContainer = document.getElementById('simulation');
  if (!simulationContainer) {
    console.error('Simulation container not found');
    return;
  }

  // Create simulation
  const simulation = new Simulation();

  // Create data recorder
  const dataRecorder = new DataRecorder(simulation);
  dataRecorder.start();

  // Create simulation persistence manager
  const simulationPersistence = new SimulationPersistence(simulation);

  // Create renderer
  const renderer = new Renderer(simulation, simulationContainer);

  // Create controls container
  const controlsContainer = document.createElement('div');
  controlsContainer.className = 'controls';
  controlsContainer.style.position = 'absolute';
  controlsContainer.style.top = '10px';
  controlsContainer.style.left = '10px';
  controlsContainer.style.zIndex = '1000';

  // Simulation controls
  const startBtn = document.createElement('button');
  startBtn.textContent = 'Start';
  startBtn.addEventListener('click', () => simulation.start());

  const pauseBtn = document.createElement('button');
  pauseBtn.textContent = 'Pause';
  pauseBtn.addEventListener('click', () => simulation.pause());

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset';
  resetBtn.addEventListener('click', () => simulation.reset());

  // Analytics dashboard
  const analyticsBtn = document.createElement('button');
  analyticsBtn.textContent = 'Open Analytics';
  analyticsBtn.addEventListener('click', () => {
    window.open('analytics.html', '_blank');
  });

  // Persistence controls
  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save';
  saveBtn.addEventListener('click', () => {
    if (simulationPersistence.saveToLocalStorage()) {
      alert('Simulation saved!');
    }
  });

  const exportBtn = document.createElement('button');
  exportBtn.textContent = 'Export';
  exportBtn.addEventListener('click', () => {
    simulationPersistence.exportToFile();
  });

  const loadBtn = document.createElement('button');
  loadBtn.textContent = 'Load';
  loadBtn.addEventListener('click', () => {
    if (confirm('This will replace the current simulation. Continue?')) {
      const success = simulationPersistence.loadFromLocalStorage();
      if (success) {
        alert('Simulation loaded!');
      } else {
        alert('No saved simulation found');
      }
    }
  });

  // File input for importing
  const importInput = document.createElement('input');
  importInput.type = 'file';
  importInput.accept = '.json';
  importInput.style.display = 'none';
  document.body.appendChild(importInput);

  importInput.addEventListener('change', (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      simulationPersistence
        .importFromFile(file)
        .then(() => {
          alert('Simulation imported!');
        })
        .catch((error) => {
          alert('Error importing simulation: ' + error.message);
        });
    }
  });

  const importBtn = document.createElement('button');
  importBtn.textContent = 'Import';
  importBtn.addEventListener('click', () => {
    importInput.click();
  });

  // Add buttons to container
  controlsContainer.appendChild(startBtn);
  controlsContainer.appendChild(pauseBtn);
  controlsContainer.appendChild(resetBtn);
  controlsContainer.appendChild(document.createElement('span')).textContent =
    ' | ';
  controlsContainer.appendChild(analyticsBtn);
  controlsContainer.appendChild(document.createElement('span')).textContent =
    ' | ';
  controlsContainer.appendChild(saveBtn);
  controlsContainer.appendChild(exportBtn);
  controlsContainer.appendChild(loadBtn);
  controlsContainer.appendChild(importBtn);

  document.body.appendChild(controlsContainer);

  // Initialize simulation
  simulation.reset();
});
