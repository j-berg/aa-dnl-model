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
            # Amino acids (hepatic portal vein concentrations)
            # PMID: 15572415 - Measured in portal vein
            'alanine': Metabolite('alanine', 0.0),
            'glutamate': Metabolite('glutamate', 0.0),
            'aspartate': Metabolite('aspartate', 0.0),
            
            # Early metabolic intermediates
            # PMID: 19860672 - Liver concentrations
            'pyruvate': Metabolite('pyruvate', 0.0),
            'acetyl_coa': Metabolite('acetyl_coa', 0.0),
            
            # Fatty acid synthesis intermediates
            # PMID: 27207583 - Hepatic lipogenic pathway
            'malonyl_coa': Metabolite('malonyl_coa', 0.0),
            'acetoacetyl_coa': Metabolite('acetoacetyl_coa', 0.0),  # C4
            'butyryl_coa': Metabolite('butyryl_coa', 0.0),          # C4
            'hexanoyl_coa': Metabolite('hexanoyl_coa', 0.0),        # C6
            'octanoyl_coa': Metabolite('octanoyl_coa', 0.0),        # C8
            'decanoyl_coa': Metabolite('decanoyl_coa', 0.0),        # C10
            'lauroyl_coa': Metabolite('lauroyl_coa', 0.0),          # C12
            'myristoyl_coa': Metabolite('myristoyl_coa', 0.0),      # C14
            'palmitoyl_coa': Metabolite('palmitoyl_coa', 0.0),      # C16
            'palmitate': Metabolite('palmitate', 0.0)               # Free fatty acid
        }
        
        # Define reactions
        self.reactions = [
            # Amino acid catabolism
            # PMID: 17696492 - Hepatic amino acid catabolism
            Reaction('alanine_to_pyruvate',
                    [self.metabolites['alanine']],
                    [self.metabolites['pyruvate']],
                    0.15),
            
            Reaction('glutamate_to_pyruvate',
                    [self.metabolites['glutamate']],
                    [self.metabolites['pyruvate']],
                    0.14),
            
            Reaction('aspartate_to_pyruvate',
                    [self.metabolites['aspartate']],
                    [self.metabolites['pyruvate']],
                    0.13),
            
            # Pyruvate to Acetyl-CoA
            Reaction('pyruvate_to_acetyl_coa',
                    [self.metabolites['pyruvate']],
                    [self.metabolites['acetyl_coa']],
                    0.20),
            
            # Initial steps of lipogenesis
            # PMID: 23892683 - Lipogenic enzyme kinetics
            Reaction('acetyl_coa_to_malonyl_coa',
                    [self.metabolites['acetyl_coa']],
                    [self.metabolites['malonyl_coa']],
                    0.12),
            
            # Fatty acid synthesis steps
            # PMID: 27207583 - FAS complex
            Reaction('initial_condensation',
                    [self.metabolites['acetyl_coa'], self.metabolites['malonyl_coa']],
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
            
            # Final release of free fatty acid
            # PMID: 25784669 - Thioesterase
            Reaction('palmitoyl_to_palmitate',
                    [self.metabolites['palmitoyl_coa']],
                    [self.metabolites['palmitate']],
                    0.09)
        ]
        
        self.time_points = []
        self.concentration_history = {met: [] for met in self.metabolites}

    def set_initial_concentrations(self, scenario: str):
        """Set initial concentrations based on scenario"""
        # Define default initial concentrations
        initial_concentrations = {
            'alanine': 0.0,
            'glutamate': 0.0,
            'aspartate': 0.0,
            'pyruvate': 0.1,
            'acetyl_coa': 0.14,
            'malonyl_coa': 0.02,
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
            # Modify only the amino acid concentrations for SMM-based scenario
            initial_concentrations.update({
                'alanine': 0.8,      # ~1.6x normal (0.5 mM baseline)
                'glutamate': 0.96,   # ~1.6x normal (0.6 mM baseline)
                'aspartate': 0.08,   # ~1.6x normal (0.05 mM baseline)
            })
        elif scenario == "conventional":  # 200g protein intake
            # Modify only the amino acid concentrations for conventional scenario
            initial_concentrations.update({
                'alanine': 1.25,     # ~2.5x normal
                'glutamate': 1.5,    # ~2.5x normal
                'aspartate': 0.125,  # ~2.5x normal
            })
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        for met_name, conc in initial_concentrations.items():
            self.metabolites[met_name].concentration = conc

    def calculate_flux(self, reaction: Reaction) -> float:
        """Calculate reaction flux based on mass action kinetics"""
        flux = reaction.rate_constant
        for substrate in reaction.substrates:
            flux *= substrate.concentration
        return flux

    def simulate(self, time_points: np.ndarray):
        """Run the simulation for given time points"""
        self.time_points = time_points
        dt = time_points[1] - time_points[0]
        
        # Record initial concentrations
        for met_name, met in self.metabolites.items():
            self.concentration_history[met_name].append(met.concentration)
        
        # Simulate
        for t in time_points[1:]:
            # Calculate all fluxes
            fluxes = [self.calculate_flux(rxn) for rxn in self.reactions]
            
            # Update concentrations
            for met_name, met in self.metabolites.items():
                delta_conc = 0
                for rxn, flux in zip(self.reactions, fluxes):
                    if met in rxn.products:
                        delta_conc += flux
                    if met in rxn.substrates:
                        delta_conc -= flux
                
                met.concentration += delta_conc * dt
                self.concentration_history[met_name].append(met.concentration)

def run_comparison():
    # Create time points (6 hours = 360 minutes)
    t = np.linspace(0, 360, 1000)
    
    # SMM-based protein intake (130g)
    network_smm = MetabolicNetwork()
    network_smm.set_initial_concentrations("smm_based")
    network_smm.simulate(t)
    
    # Conventional protein intake (200g)
    network_conv = MetabolicNetwork()
    network_conv.set_initial_concentrations("conventional")
    network_conv.simulate(t)
    
    # Plot comparisons
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define colors for metabolites
    color_dict = {
        'alanine': 'blue',
        'glutamate': 'green',
        'aspartate': 'red',
        'acetyl_coa': 'purple',
        'malonyl_coa': 'orange',
        'butyryl_coa': 'cyan',
        'octanoyl_coa': 'magenta',
        'lauroyl_coa': 'brown',
        'palmitoyl_coa': 'olive',
        'palmitate': 'black'
    }
    
    # Plot amino acids
    for met_name in ['alanine', 'glutamate', 'aspartate']:
        color = color_dict[met_name]
        axes[0,0].plot(t/60, network_smm.concentration_history[met_name], 
                      label=met_name, color=color)
        axes[0,0].plot(t/60, network_conv.concentration_history[met_name], 
                      linestyle='--', color=color)
    axes[0,0].set_title('Amino Acids')
    axes[0,0].set_xlabel('Time (hours)')
    axes[0,0].set_ylabel('Concentration (mM)')
    # Add scenario legend
    axes[0,0].plot([], [], color='gray', label='130g protein', linestyle='-')
    axes[0,0].plot([], [], color='gray', label='200g protein', linestyle='--')
    axes[0,0].legend(ncol=2)
    
    # Plot early lipogenesis intermediates
    for met_name in ['acetyl_coa', 'malonyl_coa']:
        color = color_dict[met_name]
        axes[0,1].plot(t/60, network_smm.concentration_history[met_name], 
                      label=met_name, color=color)
        axes[0,1].plot(t/60, network_conv.concentration_history[met_name], 
                      linestyle='--', color=color)
    axes[0,1].set_title('Early Lipogenesis Intermediates')
    axes[0,1].set_xlabel('Time (hours)')
    axes[0,1].set_ylabel('Concentration (mM)')
    axes[0,1].legend(ncol=2)
    
    # Plot FAS intermediates
    fas_intermediates = ['butyryl_coa', 'octanoyl_coa', 'lauroyl_coa', 'palmitoyl_coa']
    for met_name in fas_intermediates:
        color = color_dict[met_name]
        label = f'C{len(met_name.split("_")[0])-2}'  # Convert name to carbon number
        axes[1,0].plot(t/60, network_smm.concentration_history[met_name], 
                      label=label, color=color)
        axes[1,0].plot(t/60, network_conv.concentration_history[met_name], 
                      linestyle='--', color=color)
    axes[1,0].set_title('Fatty Acid Synthesis Intermediates')
    axes[1,0].set_xlabel('Time (hours)')
    axes[1,0].set_ylabel('Concentration (mM)')
    axes[1,0].legend(ncol=2)
    
    # Plot final product (palmitate)
    color = color_dict['palmitate']
    axes[1,1].plot(t, network_smm.concentration_history['palmitate'], 
                  label='palmitate', color=color)
    axes[1,1].plot(t, network_conv.concentration_history['palmitate'], 
                  linestyle='--', color=color)
    axes[1,1].set_title('Palmitate Production')
    # Add scenario legend
    axes[1,1].plot([], [], color='gray', label='Normal', linestyle='-')
    axes[1,1].plot([], [], color='gray', label='Excess', linestyle='--')
    axes[1,1].legend(ncol=2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()