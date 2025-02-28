// src/analytics/dashboard.ts
import { Simulation } from '../core/sim';
import { Agent } from '../entities/agent';
import { Species } from '../evolution/species';
import { FoodType } from '../entities/food';
import config from '../cfg';
import { Vector2D } from '../utils/math';

export class AnalyticsDashboard {
  private simulation: Simulation;
  private container: HTMLElement;
  private dashboardElement!: HTMLElement;
  private selectedAgentId: number | null = null;
  private generationStats: Map<number, GenerationStats> = new Map();
  private agentActivityLog: Map<number, AgentActivity[]> = new Map();

  // Charts and visualizations
  private populationChart: any; // We'll use a charting library later
  private fitnessChart: any;
  private speciesChart: any;

  constructor(simulation: Simulation, container: HTMLElement) {
    this.simulation = simulation;
    this.container = container;
    this.initDashboard();

    // Force visibility after initialization
    this.dashboardElement.style.display = 'block';
  }

  private initDashboard(): void {
    // Create dashboard container
    this.dashboardElement = document.createElement('div');
    this.dashboardElement.className = 'analytics-dashboard';
    this.dashboardElement.style.position = 'absolute';
    this.dashboardElement.style.right = '0';
    this.dashboardElement.style.top = '0';
    this.dashboardElement.style.width = '300px';
    this.dashboardElement.style.height = '100%';
    this.dashboardElement.style.backgroundColor = 'rgba(0, 10, 20, 0.85)';
    this.dashboardElement.style.color = '#fff';
    this.dashboardElement.style.padding = '10px';
    this.dashboardElement.style.overflowY = 'auto';
    this.dashboardElement.style.fontFamily = 'monospace';
    this.dashboardElement.style.fontSize = '12px';
    this.dashboardElement.style.boxShadow = '-2px 0 10px rgba(0, 0, 0, 0.5)';
    this.dashboardElement.style.zIndex = '1000';

    this.container.appendChild(this.dashboardElement);

    // Create tabs
    this.createTabs();

    // Set up update loop
    setInterval(() => this.update(), 500); // Update every 500ms
  }

  private createTabs(): void {
    const tabs = [
      { id: 'overview', label: 'Overview' },
      { id: 'agents', label: 'Agents' },
      { id: 'species', label: 'Species' },
      { id: 'evolution', label: 'Evolution' },
      { id: 'neural', label: 'Neural Networks' },
      { id: 'environment', label: 'Environment' },
    ];

    const tabsContainer = document.createElement('div');
    tabsContainer.className = 'tabs';
    tabsContainer.style.display = 'flex';
    tabsContainer.style.marginBottom = '10px';
    tabsContainer.style.borderBottom = '1px solid #555';

    tabs.forEach((tab) => {
      const tabElement = document.createElement('div');
      tabElement.className = 'tab';
      tabElement.textContent = tab.label;
      tabElement.dataset.tabId = tab.id;
      tabElement.style.padding = '5px 10px';
      tabElement.style.cursor = 'pointer';
      tabElement.style.borderRight = '1px solid #555';

      tabElement.addEventListener('click', () => this.showTab(tab.id));

      tabsContainer.appendChild(tabElement);
    });

    this.dashboardElement.appendChild(tabsContainer);

    // Create content containers for each tab
    tabs.forEach((tab) => {
      const contentElement = document.createElement('div');
      contentElement.className = 'tab-content';
      contentElement.id = `tab-${tab.id}`;
      contentElement.style.display = 'none';

      this.dashboardElement.appendChild(contentElement);
    });

    // Show first tab by default
    this.showTab('overview');
  }

  private showTab(tabId: string): void {
    // Hide all tabs
    const tabContents = this.dashboardElement.querySelectorAll('.tab-content');
    tabContents.forEach((el) => ((el as HTMLElement).style.display = 'none'));

    // Show selected tab
    const selectedTab = document.getElementById(`tab-${tabId}`);
    if (selectedTab) {
      selectedTab.style.display = 'block';
    }

    // Highlight selected tab
    const tabs = this.dashboardElement.querySelectorAll('.tab');
    tabs.forEach((tab) => {
      (tab as HTMLElement).style.backgroundColor =
        tab.getAttribute('data-tab-id') === tabId ? '#334' : 'transparent';
    });
  }

  public update(): void {
    // Collect current data
    this.collectGenerationStats();
    this.collectAgentActivities();

    // Update each tab content
    this.updateOverviewTab();
    this.updateAgentsTab();
    this.updateSpeciesTab();
    this.updateEvolutionTab();
    this.updateNeuralNetworkTab();
    this.updateEnvironmentTab();
  }

  private collectGenerationStats(): void {
    const generation = this.simulation.getGeneration();
    const agents = this.simulation.getAgents();

    if (!this.generationStats.has(generation)) {
      this.generationStats.set(generation, {
        generation,
        maxFitness: 0,
        avgFitness: 0,
        speciesCounts: {},
        mutationRates: [],
        foodConsumed: {
          [FoodType.BASIC]: 0,
          [FoodType.SUPER]: 0,
          [FoodType.POISON]: 0,
        },
        deaths: 0,
        births: 0,
      });
    }

    const stats = this.generationStats.get(generation)!;

    // Update fitness stats
    let totalFitness = 0;
    let maxFitness = 0;

    agents.forEach((agent) => {
      totalFitness += agent.fitness;
      maxFitness = Math.max(maxFitness, agent.fitness);

      // Track mutation rates
      if (!stats.mutationRates.includes(agent.genome.mutationRate)) {
        stats.mutationRates.push(agent.genome.mutationRate);
      }

      // Track species distribution
      if (agent.species) {
        const speciesId = agent.species.id;
        stats.speciesCounts[speciesId] =
          (stats.speciesCounts[speciesId] || 0) + 1;
      }
    });

    stats.avgFitness = agents.length > 0 ? totalFitness / agents.length : 0;
    stats.maxFitness = Math.max(stats.maxFitness, maxFitness);

    // Update deaths
    const deadAgents = this.simulation
      .getLineageTracker()
      .getGenerationRecords(generation)
      .filter((r) => r.deathTick !== null);
    stats.deaths = deadAgents.length;

    // Update births
    stats.births = this.simulation
      .getLineageTracker()
      .getGenerationRecords(generation).length;
  }

  private collectAgentActivities(): void {
    const agents = this.simulation.getAgents();
    const tick = this.simulation.getTickCount();

    agents.forEach((agent) => {
      if (!this.agentActivityLog.has(agent.id)) {
        this.agentActivityLog.set(agent.id, []);
      }

      const log = this.agentActivityLog.get(agent.id)!;

      // Only log every 10 ticks to avoid too much data
      if (tick % 10 === 0) {
        log.push({
          tick,
          position: { ...agent.position },
          energy: agent.energy,
          fitness: agent.fitness,
          age: agent.age,
          velocity: { ...agent.velocity },
          // We could add more detailed activity data here
        });

        // Keep log size manageable
        if (log.length > 100) {
          log.shift();
        }
      }
    });
  }

  private updateOverviewTab(): void {
    const overviewTab = document.getElementById('tab-overview');
    if (!overviewTab) return;

    const generation = this.simulation.getGeneration();
    const tick = this.simulation.getTickCount();
    const agents = this.simulation.getAgents();
    const species = this.simulation.getSpeciesManager().getActiveSpecies();

    // Calculate stats
    const totalFitness = agents.reduce((sum, agent) => sum + agent.fitness, 0);
    const avgFitness = agents.length > 0 ? totalFitness / agents.length : 0;
    const maxFitness =
      agents.length > 0 ? Math.max(...agents.map((a) => a.fitness)) : 0;

    const avgMutationRate =
      agents.length > 0
        ? agents.reduce((sum, agent) => sum + agent.genome.mutationRate, 0) /
          agents.length
        : 0;

    // Create HTML content
    let html = `
      <h2>Simulation Overview</h2>
      <div class="stat-group">
        <div class="stat"><span>Generation:</span> ${generation}</div>
        <div class="stat"><span>Tick:</span> ${tick}</div>
        <div class="stat"><span>Agents:</span> ${agents.length}</div>
        <div class="stat"><span>Species:</span> ${species.length}</div>
      </div>
      
      <h3>Fitness</h3>
      <div class="stat-group">
        <div class="stat"><span>Avg Fitness:</span> ${avgFitness.toFixed(
          2
        )}</div>
        <div class="stat"><span>Max Fitness:</span> ${maxFitness.toFixed(
          2
        )}</div>
      </div>
      
      <h3>Evolution</h3>
      <div class="stat-group">
        <div class="stat"><span>Avg Mutation Rate:</span> ${avgMutationRate.toFixed(
          4
        )}</div>
        <div class="stat"><span>Most Common Species:</span> ${
          this.getMostCommonSpecies()?.name || 'None'
        }</div>
      </div>
      
      <h3>Environment</h3>
      <div class="stat-group">
        <div class="stat"><span>Time of Day:</span> ${this.simulation
          .getEnvironmentCycle()
          .getTimeOfDay()}</div>
        <div class="stat"><span>Food Count:</span> ${
          this.simulation.getFoods().length
        }</div>
      </div>
    `;

    overviewTab.innerHTML = html;
    this.applyTabStyles(overviewTab);
  }

  private updateAgentsTab(): void {
    const agentsTab = document.getElementById('tab-agents');
    if (!agentsTab) return;

    const agents = this.simulation.getAgents();

    // Create HTML content
    let html = `
      <h2>Agents (${agents.length})</h2>
      <div class="agent-list">
    `;

    // Sort agents by fitness
    const sortedAgents = [...agents].sort((a, b) => b.fitness - a.fitness);

    // Display top agents
    sortedAgents.slice(0, 10).forEach((agent) => {
      const speciesName = agent.species ? agent.species.name : 'Unknown';
      const speciesColor = agent.species ? agent.species.color : '#888';

      html += `
        <div class="agent-item" data-agent-id="${agent.id}">
          <div class="agent-color" style="background-color: ${speciesColor}"></div>
          <div class="agent-info">
            <div class="agent-id">ID: ${agent.id}</div>
            <div class="agent-stats">
              Fitness: ${agent.fitness.toFixed(2)} | 
              Energy: ${agent.energy.toFixed(2)} | 
              Age: ${agent.age}
            </div>
            <div class="agent-species">Species: ${speciesName}</div>
          </div>
        </div>
      `;
    });

    html += `</div>`;

    // Add agent detail view if an agent is selected
    if (this.selectedAgentId !== null) {
      const selectedAgent = agents.find((a) => a.id === this.selectedAgentId);
      if (selectedAgent) {
        html += this.renderAgentDetail(selectedAgent);
      } else {
        this.selectedAgentId = null;
      }
    }

    agentsTab.innerHTML = html;
    this.applyTabStyles(agentsTab);

    // Add click event listeners to agent items
    const agentItems = agentsTab.querySelectorAll('.agent-item');
    agentItems.forEach((item) => {
      item.addEventListener('click', () => {
        const agentId = Number(item.getAttribute('data-agent-id'));
        this.selectedAgentId = agentId;
        this.updateAgentsTab();
      });
    });
  }

  private renderAgentDetail(agent: Agent): string {
    // Get agent activity log
    const activityLog = this.agentActivityLog.get(agent.id) || [];

    // Get agent ancestry
    const ancestors = this.simulation
      .getLineageTracker()
      .getAncestors(agent.id);

    return `
      <div class="agent-detail">
        <h3>Agent ${agent.id} Details</h3>
        
        <div class="detail-section">
          <h4>Basic Information</h4>
          <div class="stat-group">
            <div class="stat"><span>Species:</span> ${
              agent.species?.name || 'Unknown'
            }</div>
            <div class="stat"><span>Mutation Rate:</span> ${agent.genome.mutationRate.toFixed(
              4
            )}</div>
            <div class="stat"><span>Position:</span> (${agent.position.x.toFixed(
              1
            )}, ${agent.position.y.toFixed(1)})</div>
            <div class="stat"><span>Velocity:</span> (${agent.velocity.x.toFixed(
              2
            )}, ${agent.velocity.y.toFixed(2)})</div>
            <div class="stat"><span>Metabolism:</span> ${agent.metabolismMultiplier.toFixed(
              2
            )}x</div>
          </div>
        </div>
        
        <div class="detail-section">
          <h4>Ancestry</h4>
          <div class="ancestry">
            ${
              ancestors.length > 0
                ? ancestors
                    .slice(0, 5)
                    .map((a) => `<div>ID: ${a.id} (Gen ${a.generation})</div>`)
                    .join('')
                : 'First generation'
            }
            ${
              ancestors.length > 5
                ? `<div>... and ${ancestors.length - 5} more</div>`
                : ''
            }
          </div>
        </div>
        
        <div class="detail-section">
          <h4>Recent Activity</h4>
          <div class="activity-log">
            ${activityLog
              .slice(-5)
              .map(
                (entry) =>
                  `<div>Tick ${entry.tick}: Energy=${entry.energy.toFixed(
                    1
                  )}, Fitness=${entry.fitness.toFixed(1)}</div>`
              )
              .join('')}
          </div>
        </div>
        
        <div class="detail-section">
          <h4>Neural Network</h4>
          <div class="neural-network-viz">
            <div>Input Neurons: ${config.inputNeurons}</div>
            <div>Hidden Neurons: ${config.hiddenNeurons.join(', ')}</div>
            <div>Output Neurons: ${config.outputNeurons}</div>
            <div>Total Weights: ${agent.genome.weights.length}</div>
          </div>
        </div>
      </div>
    `;
  }

  private updateSpeciesTab(): void {
    const speciesTab = document.getElementById('tab-species');
    if (!speciesTab) return;

    const allSpecies = this.simulation.getSpeciesManager().getSpecies();
    const activeSpecies = this.simulation
      .getSpeciesManager()
      .getActiveSpecies();

    // Create HTML content
    let html = `
      <h2>Species (${activeSpecies.length} active, ${allSpecies.length} total)</h2>
      <div class="species-list">
    `;

    // Sort species by number of members
    const sortedSpecies = [...activeSpecies].sort(
      (a, b) => b.members.length - a.members.length
    );

    sortedSpecies.forEach((species) => {
      html += `
        <div class="species-item">
          <div class="species-color" style="background-color: ${
            species.color
          }"></div>
          <div class="species-info">
            <div class="species-name">${species.name}</div>
            <div class="species-stats">
              Members: ${species.members.length} | 
              Best Fitness: ${species.bestFitness.toFixed(2)}
            </div>
            <div class="species-history">
              Created: Gen ${species.creationGeneration} | 
              Age: ${
                this.simulation.getGeneration() - species.creationGeneration
              } generations
            </div>
          </div>
        </div>
      `;
    });

    html += `</div>`;

    // Add species diversity chart placeholder
    html += `
      <div class="chart-section">
        <h3>Species Diversity Over Time</h3>
        <div id="species-diversity-chart" class="chart-placeholder">
          [Species Diversity Chart - Will be implemented]
        </div>
      </div>
    `;

    speciesTab.innerHTML = html;
    this.applyTabStyles(speciesTab);
  }

  private updateEvolutionTab(): void {
    const evolutionTab = document.getElementById('tab-evolution');
    if (!evolutionTab) return;

    // Get all generation stats
    const stats = Array.from(this.generationStats.values()).sort(
      (a, b) => a.generation - b.generation
    );

    // Create HTML content
    let html = `
      <h2>Evolution Metrics</h2>
      
      <div class="chart-section">
        <h3>Fitness Progression</h3>
        <div id="fitness-chart" class="chart-placeholder">
          [Fitness Chart - Will be implemented]
        </div>
      </div>
      
      <h3>Generation History</h3>
      <div class="generation-history">
        <table>
          <thead>
            <tr>
              <th>Gen</th>
              <th>Avg Fitness</th>
              <th>Max Fitness</th>
              <th>Species</th>
              <th>Births/Deaths</th>
            </tr>
          </thead>
          <tbody>
    `;

    stats.slice(-10).forEach((stat) => {
      const speciesCount = Object.keys(stat.speciesCounts).length;

      html += `
        <tr>
          <td>${stat.generation}</td>
          <td>${stat.avgFitness.toFixed(2)}</td>
          <td>${stat.maxFitness.toFixed(2)}</td>
          <td>${speciesCount}</td>
          <td>${stat.births}/${stat.deaths}</td>
        </tr>
      `;
    });

    html += `
          </tbody>
        </table>
      </div>
      
      <h3>Mutation Rates</h3>
      <div class="mutation-rates">
        Current Range: ${this.getCurrentMutationRateRange()}
      </div>
    `;

    evolutionTab.innerHTML = html;
    this.applyTabStyles(evolutionTab);
  }

  private updateNeuralNetworkTab(): void {
    const neuralTab = document.getElementById('tab-neural');
    if (!neuralTab) return;

    const agents = this.simulation.getAgents();

    // Create HTML content
    let html = `
      <h2>Neural Network Analysis</h2>
      
      <h3>Select an agent to visualize its neural network</h3>
      <select id="agent-select">
        <option value="">--Select Agent--</option>
    `;

    // Sort agents by fitness
    const sortedAgents = [...agents].sort((a, b) => b.fitness - a.fitness);

    sortedAgents.slice(0, 20).forEach((agent) => {
      html += `<option value="${agent.id}">Agent ${
        agent.id
      } (Fitness: ${agent.fitness.toFixed(2)})</option>`;
    });

    html += `</select>`;

    // Placeholder for neural network visualization
    html += `
      <div id="nn-visualization" class="neural-visualization">
        <p>Select an agent to view its neural network</p>
      </div>
      
      <h3>Weight Distribution Analysis</h3>
      <div class="weight-distribution">
        ${this.renderWeightDistributionStats()}
      </div>
    `;

    neuralTab.innerHTML = html;
    this.applyTabStyles(neuralTab);

    // Add event listener to agent select dropdown
    const agentSelect = neuralTab.querySelector('#agent-select');
    if (agentSelect) {
      agentSelect.addEventListener('change', (e) => {
        const agentId = Number((e.target as HTMLSelectElement).value);
        const selectedAgent = agents.find((a) => a.id === agentId);

        if (selectedAgent) {
          const nnViz = neuralTab.querySelector('#nn-visualization');
          if (nnViz) {
            nnViz.innerHTML =
              this.renderNeuralNetworkVisualization(selectedAgent);
          }
        }
      });
    }
  }

  private renderWeightDistributionStats(): string {
    const agents = this.simulation.getAgents();

    // Skip if no agents
    if (agents.length === 0) {
      return '<p>No agents to analyze</p>';
    }

    // Collect all weights across all agents
    let allWeights: number[] = [];
    agents.forEach((agent) => {
      allWeights = allWeights.concat(agent.genome.weights);
    });

    // Calculate weight distribution stats
    const min = Math.min(...allWeights);
    const max = Math.max(...allWeights);
    const avg = allWeights.reduce((sum, w) => sum + w, 0) / allWeights.length;
    const sorted = [...allWeights].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];

    // Count weights in ranges
    const ranges = {
      negative: allWeights.filter((w) => w < -0.5).length,
      smallNegative: allWeights.filter((w) => w >= -0.5 && w < 0).length,
      zero: allWeights.filter((w) => w === 0).length,
      smallPositive: allWeights.filter((w) => w > 0 && w <= 0.5).length,
      positive: allWeights.filter((w) => w > 0.5).length,
    };

    return `
      <div class="stat-group">
        <div class="stat"><span>Total Weights:</span> ${allWeights.length}</div>
        <div class="stat"><span>Min:</span> ${min.toFixed(3)}</div>
        <div class="stat"><span>Max:</span> ${max.toFixed(3)}</div>
        <div class="stat"><span>Avg:</span> ${avg.toFixed(3)}</div>
        <div class="stat"><span>Median:</span> ${median.toFixed(3)}</div>
      </div>
      
      <h4>Weight Distribution:</h4>
      <div class="weight-bars">
        <div class="weight-bar">
          <div class="bar-label">< -0.5</div>
          <div class="bar" style="width: ${
            (ranges.negative / allWeights.length) * 100
          }%"></div>
          <div class="bar-value">${Math.round(
            (ranges.negative / allWeights.length) * 100
          )}%</div>
        </div>
        <div class="weight-bar">
          <div class="bar-label">-0.5 to 0</div>
          <div class="bar" style="width: ${
            (ranges.smallNegative / allWeights.length) * 100
          }%"></div>
          <div class="bar-value">${Math.round(
            (ranges.smallNegative / allWeights.length) * 100
          )}%</div>
        </div>
        <div class="weight-bar">
          <div class="bar-label">0</div>
          <div class="bar" style="width: ${
            (ranges.zero / allWeights.length) * 100
          }%"></div>
          <div class="bar-value">${Math.round(
            (ranges.zero / allWeights.length) * 100
          )}%</div>
        </div>
        <div class="weight-bar">
          <div class="bar-label">0 to 0.5</div>
          <div class="bar" style="width: ${
            (ranges.smallPositive / allWeights.length) * 100
          }%"></div>
          <div class="bar-value">${Math.round(
            (ranges.smallPositive / allWeights.length) * 100
          )}%</div>
        </div>
        <div class="weight-bar">
          <div class="bar-label">> 0.5</div>
          <div class="bar" style="width: ${
            (ranges.positive / allWeights.length) * 100
          }%"></div>
          <div class="bar-value">${Math.round(
            (ranges.positive / allWeights.length) * 100
          )}%</div>
        </div>
      </div>
    `;
  }

  private renderNeuralNetworkVisualization(agent: Agent): string {
    // This is a simplified visualization - you'd implement a more
    // detailed visualization with actual neural connections
    return `
      <h4>Neural Network for Agent ${agent.id}</h4>
      <div class="neural-layers">
        <div class="layer">
          <div class="layer-name">Input Layer</div>
          ${Array(config.inputNeurons)
            .fill(0)
            .map((_, i) => `<div class="neuron">I${i}</div>`)
            .join('')}
        </div>
        
        ${config.hiddenNeurons
          .map(
            (count, layerIdx) => `
          <div class="layer">
            <div class="layer-name">Hidden Layer ${layerIdx + 1}</div>
            ${Array(count)
              .fill(0)
              .map((_, i) => `<div class="neuron">H${layerIdx + 1}-${i}</div>`)
              .join('')}
          </div>
        `
          )
          .join('')}
        
        <div class="layer">
          <div class="layer-name">Output Layer</div>
          ${Array(config.outputNeurons)
            .fill(0)
            .map(
              (_, i) =>
                `<div class="neuron ${
                  i === 0
                    ? 'neuron-forward'
                    : i === 1
                    ? 'neuron-turn'
                    : 'neuron-speed'
                }">
              O${i}
            </div>`
            )
            .join('')}
        </div>
      </div>
      
      <div class="network-stats">
        <h4>Connection Strengths</h4>
        <p>This agent has ${agent.genome.weights.length} connections.</p>
        <p>Strongest connection: ${Math.max(
          ...agent.genome.weights.map(Math.abs)
        ).toFixed(3)}</p>
        <p>Average strength: ${(
          agent.genome.weights.reduce((sum, w) => sum + Math.abs(w), 0) /
          agent.genome.weights.length
        ).toFixed(3)}</p>
      </div>
    `;
  }

  private updateEnvironmentTab(): void {
    const environmentTab = document.getElementById('tab-environment');
    if (!environmentTab) return;

    const foods = this.simulation.getFoods();
    const foodTypes = {
      [FoodType.BASIC]: foods.filter((f) => f.type === FoodType.BASIC).length,
      [FoodType.SUPER]: foods.filter((f) => f.type === FoodType.SUPER).length,
      [FoodType.POISON]: foods.filter((f) => f.type === FoodType.POISON).length,
    };

    const zones = this.simulation.getZones();

    // Calculate agent distribution by zone
    const agents = this.simulation.getAgents();
    const agentsByZone: Record<string, number> = {};

    zones.forEach((zone) => {
      const zoneAgents = agents.filter((agent) =>
        zone.contains(agent.position)
      );
      agentsByZone[zone.type] = zoneAgents.length;
    });

    // Count agents not in any zone
    const agentsOutsideZones = agents.filter(
      (agent) => !zones.some((zone) => zone.contains(agent.position))
    ).length;

    agentsByZone['outside'] = agentsOutsideZones;

    // Create HTML content
    let html = `
      <h2>Environment Analysis</h2>
      
      <h3>Food Distribution</h3>
      <div class="food-distribution">
        <div class="stat-group">
          <div class="stat"><span>Basic Food:</span> ${
            foodTypes[FoodType.BASIC]
          }</div>
          <div class="stat"><span>Super Food:</span> ${
            foodTypes[FoodType.SUPER]
          }</div>
          <div class="stat"><span>Poison:</span> ${
            foodTypes[FoodType.POISON]
          }</div>
        </div>
        
        <div class="food-chart">
          <div class="food-bar basic" style="width: ${
            (foodTypes[FoodType.BASIC] / foods.length) * 100
          }%"></div>
          <div class="food-bar super" style="width: ${
            (foodTypes[FoodType.SUPER] / foods.length) * 100
          }%"></div>
          <div class="food-bar poison" style="width: ${
            (foodTypes[FoodType.POISON] / foods.length) * 100
          }%"></div>
        </div>
      </div>
      
      <h3>Zone Population</h3>
      <div class="zone-distribution">
        <div class="stat-group">
          ${Object.entries(agentsByZone)
            .map(
              ([zone, count]) =>
                `<div class="stat"><span>${zone}:</span> ${count} agents (${Math.round(
                  (count / agents.length) * 100
                )}%)</div>`
            )
            .join('')}
        </div>
      </div>
      
      <h3>Day/Night Activity</h3>
      <div class="day-night">
        <div class="stat-group">
          <div class="stat"><span>Current:</span> ${this.simulation
            .getEnvironmentCycle()
            .getTimeOfDay()}</div>
          <div class="stat"><span>Visibility:</span> ${this.simulation
            .getEnvironmentCycle()
            .getVisibilityMultiplier()
            .toFixed(2)}x</div>
          <div class="stat"><span>Movement:</span> ${this.simulation
            .getEnvironmentCycle()
            .getMovementMultiplier()
            .toFixed(2)}x</div>
        </div>
      </div>
    `;

    environmentTab.innerHTML = html;
    this.applyTabStyles(environmentTab);
  }

  private getMostCommonSpecies(): Species | null {
    const activeSpecies = this.simulation
      .getSpeciesManager()
      .getActiveSpecies();
    if (activeSpecies.length === 0) return null;

    return activeSpecies.reduce(
      (mostCommon, species) =>
        species.members.length > mostCommon.members.length
          ? species
          : mostCommon,
      activeSpecies[0]
    );
  }

  private getCurrentMutationRateRange(): string {
    const agents = this.simulation.getAgents();
    if (agents.length === 0) return 'No agents';

    const rates = agents.map((a) => a.genome.mutationRate);
    const min = Math.min(...rates);
    const max = Math.max(...rates);
    const avg = rates.reduce((sum, r) => sum + r, 0) / rates.length;

    return `${min.toFixed(4)} - ${max.toFixed(4)} (avg: ${avg.toFixed(4)})`;
  }

  private applyTabStyles(tabElement: HTMLElement): void {
    // Apply styles to stats
    const stats = tabElement.querySelectorAll('.stat');
    stats.forEach((stat) => {
      (stat as HTMLElement).style.margin = '2px 0';

      const label = stat.querySelector('span');
      if (label) {
        label.style.color = '#aaa';
        label.style.marginRight = '5px';
      }
    });

    // Apply styles to stat groups
    const statGroups = tabElement.querySelectorAll('.stat-group');
    statGroups.forEach((group) => {
      (group as HTMLElement).style.margin = '10px 0';
      (group as HTMLElement).style.padding = '5px';
      (group as HTMLElement).style.backgroundColor = 'rgba(20, 30, 50, 0.5)';
      (group as HTMLElement).style.borderRadius = '3px';
    });

    // Style headings
    const headings = tabElement.querySelectorAll('h2, h3, h4');
    headings.forEach((heading) => {
      (heading as HTMLElement).style.borderBottom = '1px solid #555';
      (heading as HTMLElement).style.paddingBottom = '5px';
      (heading as HTMLElement).style.marginTop = '15px';
      (heading as HTMLElement).style.color = '#8af';
    });

    // Style agent items
    const agentItems = tabElement.querySelectorAll('.agent-item');
    agentItems.forEach((item) => {
      (item as HTMLElement).style.display = 'flex';
      (item as HTMLElement).style.margin = '5px 0';
      (item as HTMLElement).style.padding = '5px';
      (item as HTMLElement).style.backgroundColor = 'rgba(20, 30, 50, 0.5)';
      (item as HTMLElement).style.borderRadius = '3px';
      (item as HTMLElement).style.cursor = 'pointer';

      // Highlight selected agent
      if (this.selectedAgentId === Number(item.getAttribute('data-agent-id'))) {
        (item as HTMLElement).style.backgroundColor = 'rgba(40, 60, 100, 0.7)';
        (item as HTMLElement).style.border = '1px solid #8af';
      }

      // Make item highlight on hover
      (item as HTMLElement).onmouseover = () => {
        (item as HTMLElement).style.backgroundColor = 'rgba(40, 60, 100, 0.7)';
      };
      (item as HTMLElement).onmouseout = () => {
        if (
          this.selectedAgentId !== Number(item.getAttribute('data-agent-id'))
        ) {
          (item as HTMLElement).style.backgroundColor = 'rgba(20, 30, 50, 0.5)';
        }
      };
    });

    // Style agent color indicators
    const agentColors = tabElement.querySelectorAll('.agent-color');
    agentColors.forEach((color) => {
      (color as HTMLElement).style.width = '15px';
      (color as HTMLElement).style.height = '100%';
      (color as HTMLElement).style.marginRight = '10px';
      (color as HTMLElement).style.borderRadius = '3px';
    });

    // Style tables
    const tables = tabElement.querySelectorAll('table');
    tables.forEach((table) => {
      (table as HTMLElement).style.width = '100%';
      (table as HTMLElement).style.borderCollapse = 'collapse';
      (table as HTMLElement).style.marginTop = '10px';

      const cells = table.querySelectorAll('th, td');
      cells.forEach((cell) => {
        (cell as HTMLElement).style.padding = '5px';
        (cell as HTMLElement).style.textAlign = 'center';
        (cell as HTMLElement).style.border = '1px solid #555';
      });

      const headers = table.querySelectorAll('th');
      headers.forEach((header) => {
        (header as HTMLElement).style.backgroundColor =
          'rgba(40, 60, 100, 0.7)';
        (header as HTMLElement).style.color = '#fff';
      });

      // Alternate row colors
      const rows = table.querySelectorAll('tbody tr');
      rows.forEach((row, i) => {
        (row as HTMLElement).style.backgroundColor =
          i % 2 === 0 ? 'rgba(20, 30, 50, 0.5)' : 'rgba(30, 40, 60, 0.5)';
      });
    });

    // Style neural network visualization
    const neuralLayers = tabElement.querySelectorAll('.neural-layers');
    neuralLayers.forEach((layers) => {
      (layers as HTMLElement).style.display = 'flex';
      (layers as HTMLElement).style.justifyContent = 'space-between';
      (layers as HTMLElement).style.margin = '20px 0';
      (layers as HTMLElement).style.padding = '10px';
      (layers as HTMLElement).style.backgroundColor = 'rgba(20, 30, 50, 0.5)';
      (layers as HTMLElement).style.borderRadius = '5px';
    });

    const neuralLayers2 = tabElement.querySelectorAll('.layer');
    neuralLayers2.forEach((layer) => {
      (layer as HTMLElement).style.display = 'flex';
      (layer as HTMLElement).style.flexDirection = 'column';
      (layer as HTMLElement).style.alignItems = 'center';
      (layer as HTMLElement).style.margin = '0 5px';
    });

    const neurons = tabElement.querySelectorAll('.neuron');
    neurons.forEach((neuron) => {
      (neuron as HTMLElement).style.width = '40px';
      (neuron as HTMLElement).style.height = '40px';
      (neuron as HTMLElement).style.margin = '5px';
      (neuron as HTMLElement).style.borderRadius = '50%';
      (neuron as HTMLElement).style.backgroundColor = 'rgba(60, 100, 150, 0.7)';
      (neuron as HTMLElement).style.display = 'flex';
      (neuron as HTMLElement).style.justifyContent = 'center';
      (neuron as HTMLElement).style.alignItems = 'center';
      (neuron as HTMLElement).style.fontSize = '12px';
      (neuron as HTMLElement).style.color = '#fff';
    });

    // Style output neurons differently
    const outputNeurons = tabElement.querySelectorAll(
      '.neuron-forward, .neuron-turn, .neuron-speed'
    );
    outputNeurons.forEach((neuron) => {
      if (neuron.classList.contains('neuron-forward')) {
        (neuron as HTMLElement).style.backgroundColor =
          'rgba(100, 150, 60, 0.7)';
      } else if (neuron.classList.contains('neuron-turn')) {
        (neuron as HTMLElement).style.backgroundColor =
          'rgba(150, 100, 60, 0.7)';
      } else if (neuron.classList.contains('neuron-speed')) {
        (neuron as HTMLElement).style.backgroundColor =
          'rgba(100, 60, 150, 0.7)';
      }
    });

    // Style weight bars
    const weightBars = tabElement.querySelectorAll('.weight-bar');
    weightBars.forEach((bar) => {
      (bar as HTMLElement).style.display = 'flex';
      (bar as HTMLElement).style.alignItems = 'center';
      (bar as HTMLElement).style.margin = '5px 0';
    });

    const barLabels = tabElement.querySelectorAll('.bar-label');
    barLabels.forEach((label) => {
      (label as HTMLElement).style.width = '80px';
      (label as HTMLElement).style.paddingRight = '10px';
      (label as HTMLElement).style.textAlign = 'right';
      (label as HTMLElement).style.color = '#aaa';
    });

    const bars = tabElement.querySelectorAll('.bar');
    bars.forEach((bar) => {
      (bar as HTMLElement).style.height = '15px';
      (bar as HTMLElement).style.backgroundColor = 'rgba(60, 120, 180, 0.7)';
      (bar as HTMLElement).style.borderRadius = '3px';
      (bar as HTMLElement).style.minWidth = '5px';
    });

    const barValues = tabElement.querySelectorAll('.bar-value');
    barValues.forEach((value) => {
      (value as HTMLElement).style.paddingLeft = '10px';
      (value as HTMLElement).style.minWidth = '40px';
    });
  }
}

interface GenerationStats {
  generation: number;
  maxFitness: number;
  avgFitness: number;
  speciesCounts: Record<number, number>;
  mutationRates: number[];
  foodConsumed: Record<FoodType, number>;
  deaths: number;
  births: number;
}

interface AgentActivity {
  tick: number;
  position: Vector2D;
  energy: number;
  fitness: number;
  age: number;
  velocity: Vector2D;
}
