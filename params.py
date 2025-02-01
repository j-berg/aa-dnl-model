import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import matplotlib.pyplot as plt

@dataclass
class Metabolite:
    """Class to represent a metabolite in the network"""
    name: str
    concentration: float  # mM

@dataclass
class Reaction:
    """Class to represent a metabolic reaction"""
    name: str
    substrates: List[Metabolite]
    products: List[Metabolite]
    rate_constant: float  # per second

class MetabolicNetwork:
    def __init__(self):
        # Initialize metabolites
        self.metabolites = {
            # Amino acids (portal vein concentrations)
            'alanine': Metabolite('alanine', 0.0),
            'glutamate': Metabolite('glutamate', 0.0),
            'aspartate': Metabolite('aspartate', 0.0),
            
            # Early intermediates with branched acetyl-CoA pools
            'pyruvate': Metabolite('pyruvate', 0.0),
            'acetyl_coa_init': Metabolite('acetyl_coa_init', 0.0),    # For chain initiation
            'acetyl_coa_elong': Metabolite('acetyl_coa_elong', 0.0),  # For malonyl-CoA formation
            
            # Lipogenesis intermediates
            'malonyl_coa': Metabolite('malonyl_coa', 0.0),
            'acetoacetyl_coa': Metabolite('acetoacetyl_coa', 0.0),    # C4
            'butyryl_coa': Metabolite('butyryl_coa', 0.0),            # C4
            'hexanoyl_coa': Metabolite('hexanoyl_coa', 0.0),          # C6
            'octanoyl_coa': Metabolite('octanoyl_coa', 0.0),          # C8
            'decanoyl_coa': Metabolite('decanoyl_coa', 0.0),          # C10
            'lauroyl_coa': Metabolite('lauroyl_coa', 0.0),            # C12
            'myristoyl_coa': Metabolite('myristoyl_coa', 0.0),        # C14
            'palmitoyl_coa': Metabolite('palmitoyl_coa', 0.0),        # C16
            'palmitate': Metabolite('palmitate', 0.0)                  # Free fatty acid
        }
        
        # Define reactions
        self.reactions = [
            # Amino acid catabolism
            Reaction('alanine_to_pyruvate',
                    [self.metabolites['alanine']],
                    [self.metabolites['pyruvate']],
                    0.15),  # Liver-specific rate
            
            Reaction('glutamate_to_pyruvate',
                    [self.metabolites['glutamate']],
                    [self.metabolites['pyruvate']],
                    0.14),
            
            Reaction('aspartate_to_pyruvate',
                    [self.metabolites['aspartate']],
                    [self.metabolites['pyruvate']],
                    0.13),
            
            # Branched acetyl-CoA formation
            Reaction('pyruvate_to_acetyl_coa_init',
                    [self.metabolites['pyruvate']],
                    [self.metabolites['acetyl_coa_init']],
                    0.10),  # 40% to initiation
            
            Reaction('pyruvate_to_acetyl_coa_elong',
                    [self.metabolites['pyruvate']],
                    [self.metabolites['acetyl_coa_elong']],
                    0.15),  # 60% to elongation
            
            # Malonyl-CoA formation
            Reaction('acetyl_coa_to_malonyl_coa',
                    [self.metabolites['acetyl_coa_elong']],
                    [self.metabolites['malonyl_coa']],
                    0.12),
            
            # Fatty acid synthesis steps
            Reaction('initial_condensation',
                    [self.metabolites['acetyl_coa_init'], self.metabolites['malonyl_coa']],
                    [self.metabolites['acetoacetyl_coa']],
                    0.10),
            
            Reaction('acetoacetyl_to_butyryl',
                    [self.metabolites['acetoacetyl_coa']],
                    [self.metabolites['butyryl_coa']],
                    0.09),
            
            Reaction('butyryl_to_hexanoyl',
                    [self.metabolites['butyryl_coa'], self.metabolites['malonyl_coa']],
                    [self.metabolites['hexanoyl_coa']],
                    0.085),
            
            Reaction('hexanoyl_to_octanoyl',
                    [self.metabolites['hexanoyl_coa'], self.metabolites['malonyl_coa']],
                    [self.metabolites['octanoyl_coa']],
                    0.08),
            
            Reaction('octanoyl_to_decanoyl',
                    [self.metabolites['octanoyl_coa'], self.metabolites['malonyl_coa']],
                    [self.metabolites['decanoyl_coa']],
                    0.075),
            
            Reaction('decanoyl_to_lauroyl',
                    [self.metabolites['decanoyl_coa'], self.metabolites['malonyl_coa']],
                    [self.metabolites['lauroyl_coa']],
                    0.07),
            
            Reaction('lauroyl_to_myristoyl',
                    [self.metabolites['lauroyl_coa'], self.metabolites['malonyl_coa']],
                    [self.metabolites['myristoyl_coa']],
                    0.065),
            
            Reaction('myristoyl_to_palmitoyl',
                    [self.metabolites['myristoyl_coa'], self.metabolites['malonyl_coa']],
                    [self.metabolites['palmitoyl_coa']],
                    0.06),
            
            Reaction('palmitoyl_to_palmitate',
                    [self.metabolites['palmitoyl_coa']],
                    [self.metabolites['palmitate']],
                    0.09)
        ]
        
        # Initialize history tracking
        self.time_points = []
        self.concentration_history = {met: [] for met in self.metabolites}
        self.flux_history = {rxn.name: [] for rxn in self.reactions}

    def set_initial_concentrations(self, scenario: str):
        """Set initial concentrations based on scenario"""
        # Define default initial concentrations
        initial_concentrations = {
            # Base intermediates
            'pyruvate': 0.1,
            'acetyl_coa_init': 0.07,    # 40% of total acetyl-CoA
            'acetyl_coa_elong': 0.10,   # 60% of total acetyl-CoA
            'malonyl_coa': 0.02,
            
            # FAS intermediates
            'acetoacetyl_coa': 0.001,
            'butyryl_coa': 0.001,
            'hexanoyl_coa': 0.001,
            'octanoyl_coa': 0.001,
            'decanoyl_coa': 0.001,
            'lauroyl_coa': 0.001,
            'myristoyl_coa': 0.001,
            'palmitoyl_coa': 0.001,
            'palmitate': 0.001
        }
        
        if scenario == "smm_based":  # 130g protein intake
            # Add amino acid concentrations for SMM-based scenario
            initial_concentrations.update({
                'alanine': 0.8,      # ~1.6x normal (0.5 mM baseline)
                'glutamate': 0.96,   # ~1.6x normal (0.6 mM baseline)
                'aspartate': 0.08,   # ~1.6x normal (0.05 mM baseline)
            })
        elif scenario == "conventional":  # 200g protein intake
            # Add amino acid concentrations for conventional scenario
            initial_concentrations.update({
                'alanine': 1.25,     # ~2.5x normal
                'glutamate': 1.5,    # ~2.5x normal
                'aspartate': 0.125,  # ~2.5x normal
            })
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Set all concentrations
        for met_name, conc in initial_concentrations.items():
            self.metabolites[met_name].concentration = conc

    def calculate_flux(self, reaction: Reaction) -> float:
        """Calculate reaction flux based on mass action kinetics"""
        flux = reaction.rate_constant
        for substrate in reaction.substrates:
            flux *= substrate.concentration
        return flux

    def get_flux_distribution(self) -> Dict[str, float]:
        """Calculate current flux distribution"""
        return {rxn.name: self.calculate_flux(rxn) for rxn in self.reactions}

    def simulate(self, time_points: np.ndarray):
        """Run the simulation with flux tracking"""
        self.time_points = time_points
        dt = time_points[1] - time_points[0]
        
        # Record initial concentrations and fluxes
        for met_name, met in self.metabolites.items():
            self.concentration_history[met_name].append(met.concentration)
        
        fluxes = self.get_flux_distribution()
        for rxn_name, flux in fluxes.items():
            self.flux_history[rxn_name].append(flux)
        
        # Simulate
        for t in time_points[1:]:
            # Calculate all fluxes
            fluxes = self.get_flux_distribution()
            
            # Store fluxes
            for rxn_name, flux in fluxes.items():
                self.flux_history[rxn_name].append(flux)
            
            # Update concentrations
            for met_name, met in self.metabolites.items():
                delta_conc = 0
                for rxn in self.reactions:
                    flux = fluxes[rxn.name]
                    if met in rxn.products:
                        delta_conc += flux
                    if met in rxn.substrates:
                        delta_conc -= flux
                
                met.concentration += delta_conc * dt
                self.concentration_history[met_name].append(met.concentration)

def run_comparison():
    """Run and visualize the comparison between scenarios"""
    # Create time points (6 hours = 360 minutes)
    t = np.linspace(0, 360, 1000)
    
    # Run simulations
    network_smm = MetabolicNetwork()
    network_smm.set_initial_concentrations("smm_based")
    network_smm.simulate(t)
    
    network_conv = MetabolicNetwork()
    network_conv.set_initial_concentrations("conventional")
    network_conv.simulate(t)
    
    # Create plots (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Define colors for metabolites
    color_dict = {
        'alanine': 'blue',
        'glutamate': 'green',
        'aspartate': 'red',
        'acetyl_coa_init': 'purple',
        'acetyl_coa_elong': 'orange',
        'malonyl_coa': 'brown',
        'butyryl_coa': 'cyan',
        'octanoyl_coa': 'magenta',
        'lauroyl_coa': 'olive',
        'palmitoyl_coa': 'pink',
        'palmitate': 'black'
    }
    
    # 1. Amino Acids Plot
    for met_name in ['alanine', 'glutamate', 'aspartate']:
        color = color_dict[met_name]
        axes[0,0].plot(t/60, network_smm.concentration_history[met_name], 
                      label=met_name, color=color)
        axes[0,0].plot(t/60, network_conv.concentration_history[met_name], 
                      linestyle='--', color=color)
    axes[0,0].set_title('Amino Acid Concentrations')
    axes[0,0].set_xlabel('Time (hours)')
    axes[0,0].set_ylabel('Concentration (mM)')
    axes[0,0].plot([], [], color='gray', label='130g protein', linestyle='-')
    axes[0,0].plot([], [], color='gray', label='200g protein', linestyle='--')
    axes[0,0].legend(ncol=2)
    
    # 2. Acetyl-CoA Branching
    for met_name in ['acetyl_coa_init', 'acetyl_coa_elong']:
        color = color_dict[met_name]
        label = 'Initiation' if 'init' in met_name else 'Elongation'
        axes[0,1].plot(t/60, network_smm.concentration_history[met_name], 
                      label=f'{label} pool', color=color)
        axes[0,1].plot(t/60, network_conv.concentration_history[met_name], 
                      linestyle='--', color=color)
    axes[0,1].set_title('Acetyl-CoA Pool Distribution')
    axes[0,1].set_xlabel('Time (hours)')
    axes[0,1].set_ylabel('Concentration (mM)')
    axes[0,1].legend(ncol=2)
    
    # 3. Early FAS Intermediates
    early_fas = ['malonyl_coa', 'butyryl_coa']
    for met_name in early_fas:
        color = color_dict[met_name]
        axes[1,0].plot(t/60, network_smm.concentration_history[met_name], 
                      label=met_name, color=color)
        axes[1,0].plot(t/60, network_conv.concentration_history[met_name], 
                      linestyle='--', color=color)
    axes[1,0].set_title('Early FAS Intermediates')
    axes[1,0].set_xlabel('Time (hours)')
    axes[1,0].set_ylabel('Concentration (mM)')
    axes[1,0].legend(ncol=2)
    
    # 4. Late FAS Intermediates
    late_fas = ['octanoyl_coa', 'lauroyl_coa', 'palmitoyl_coa']
    for met_name in late_fas:
        color = color_dict[met_name]
        carbon_num = len(met_name.split("_")[0]) - 2  # Convert name to carbon number
        label = f'C{carbon_num}'
        axes[1,1].plot(t/60, network_smm.concentration_history[met_name], 
                      label=label, color=color)
        axes[1,1].plot(t/60, network_conv.concentration_history[met_name], 
                      linestyle='--', color=color)
    axes[1,1].set_title('Late FAS Intermediates')
    axes[1,1].set_xlabel('Time (hours)')
    axes[1,1].set_ylabel('Concentration (mM)')
    axes[1,1].legend(ncol=2)
    
    # 5. Pathway Fluxes - Early Steps
    early_reactions = ['alanine_to_pyruvate', 'pyruvate_to_acetyl_coa_init', 
                      'pyruvate_to_acetyl_coa_elong', 'acetyl_coa_to_malonyl_coa']
    for rxn_name in early_reactions:
        axes[2,0].plot(t/60, network_smm.flux_history[rxn_name], 
                      label=rxn_name.replace('_', ' '))
        axes[2,0].plot(t/60, network_conv.flux_history[rxn_name], 
                      linestyle='--')
    axes[2,0].set_title('Early Pathway Fluxes')
    axes[2,0].set_xlabel('Time (hours)')
    axes[2,0].set_ylabel('Flux (mM/s)')
    axes[2,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Final Product Accumulation and Net Flux
    # Plot palmitate concentration
    ax_palm = axes[2,1]
    color = color_dict['palmitate']
    ax_palm.plot(t/60, network_smm.concentration_history['palmitate'], 
                label='130g protein', color=color)
    ax_palm.plot(t/60, network_conv.concentration_history['palmitate'], 
                label='200g protein', linestyle='--', color=color)
    ax_palm.set_title('Palmitate Accumulation')
    ax_palm.set_xlabel('Time (hours)')
    ax_palm.set_ylabel('Concentration (mM)', color=color)
    
    # Add net flux to same plot with secondary y-axis
    ax_flux = ax_palm.twinx()
    flux_color = 'red'
    ax_flux.plot(t/60, network_smm.flux_history['palmitoyl_to_palmitate'], 
                label='130g flux', color=flux_color, alpha=0.5)
    ax_flux.plot(t/60, network_conv.flux_history['palmitoyl_to_palmitate'], 
                label='200g flux', linestyle='--', color=flux_color, alpha=0.5)
    ax_flux.set_ylabel('Net Flux (mM/s)', color=flux_color)
    
    # Combine legends
    lines_palm, labels_palm = ax_palm.get_legend_handles_labels()
    lines_flux, labels_flux = ax_flux.get_legend_handles_labels()
    ax_palm.legend(lines_palm + lines_flux, labels_palm + labels_flux, loc='center right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()