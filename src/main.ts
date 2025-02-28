// src/main.ts
import { Simulation } from './core/sim'; // Adjust path if needed
import { Renderer } from './visualization/renderer';
import { AnalyticsDashboard } from './analytics/dashboard';
import config from './cfg'; // Adjust path if needed
import { DataRecorder } from './analytics/data-recorder';

document.addEventListener('DOMContentLoaded', () => {
  console.log('Evolution Simulation initializing...');

  // Get simulation container
  const simulationContainer = document.getElementById(
    'simulation'
  ) as HTMLElement;

  const analyticsContainer = document.createElement('div');
  analyticsContainer.id = 'analytics';
  analyticsContainer.style.position = 'absolute';
  analyticsContainer.style.top = '0';
  analyticsContainer.style.right = '0';
  analyticsContainer.style.width = '300px';
  analyticsContainer.style.height = '100%';
  analyticsContainer.style.zIndex = '100';
  analyticsContainer.style.display = 'block'; // Start visible
  document.body.appendChild(analyticsContainer);

  if (!simulationContainer) {
    console.error('Simulation container not found');
    return;
  }

  // Create simulation
  const simulation = new Simulation();
  const dataRecorder = new DataRecorder(simulation);
  dataRecorder.start();
  // Create renderer
  const renderer = new Renderer(simulation, simulationContainer);

  // Set up UI controls (using existing buttons)
  const controlsContainer = document.createElement('div');
  controlsContainer.className = 'controls';
  controlsContainer.style.position = 'absolute';
  controlsContainer.style.top = '10px';
  controlsContainer.style.left = '10px';
  controlsContainer.style.zIndex = '1000';

  const startBtn = document.createElement('button');
  startBtn.textContent = 'Start';
  startBtn.addEventListener('click', () => simulation.start());

  const pauseBtn = document.createElement('button');
  pauseBtn.textContent = 'Pause';
  pauseBtn.addEventListener('click', () => simulation.pause());

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset';
  resetBtn.addEventListener('click', () => simulation.reset());

  const toggleAnalyticsBtn = document.createElement('button');
  toggleAnalyticsBtn.textContent = 'Open Analytics Dashboard';
  toggleAnalyticsBtn.addEventListener('click', () => {
    // Open analytics in a new window/tab
    window.open('analytics.html', '_blank');
  });

  controlsContainer.appendChild(startBtn);
  controlsContainer.appendChild(pauseBtn);
  controlsContainer.appendChild(resetBtn);
  controlsContainer.appendChild(toggleAnalyticsBtn);

  document.body.appendChild(controlsContainer);

  // Initialize simulation
  simulation.reset();
});
