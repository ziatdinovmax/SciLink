import sys
sys.path.append("../")
import scilink
from scilink.workflows.dft_workflow import DFTWorkflow

# Configure APIs
scilink.configure('google', '')
scilink.configure('futurehouse', '')
scilink.configure('materials_project', '')


#request = "A heterostructure consisting of a 2 layers of MoS2 (2H phase) on top of a 2 layers of WS2 (2H phase), with a 3x3 lateral cell."
#request = "3x3 Cu(100) slab, 4 layers, with a single NH3 molecule adsorbed on hollow site"
#request = "3x3 Ni(111) slab, 4 layers, with CO adsorbed at top site"
#request = "5x5 MoS2 monolayer, 2H phase, with a single Sulfur vacancy."
#request = "5x5x1 MoS2 monolayer supercell, 2H phase, with a single disulfur vacancy (VS2)"
#request = "8x3x1 MoS2 monolayer supercell, 2H phase, with a line of adjacent disulfur vacancies (VS2) along the zigzag direction"
#request = "8x3x1 MoS2 monolayer supercell, 2H phase, with a line of adjacent single sulfur vacancies (VS) along the zigzag direction"
#request = "5x5 Graphene supercell, 2D, with a single vacancy and a neighboring epoxide group"
#request = "6x6 Graphene supercell, 2D, with a single epoxide group on a C-C bond"
#request = "6x6 Graphene supercell, 2D, with a single hydroxyl group on a Carbon atom"
#request = "Rectangular 10x6 Graphene supercell, 2D, with a sinusoidal z-height modulation (wrinkle)"
#request = "a 4x4x4 supercell of BCC iron with a single vacancy at the center"
#request = "4x4x2 Gallium Nitride supercell, wurtzite phase, with a single Gallium vacancy"
#request = "4x4x2 Gallium Nitride supercell, wurtzite phase, with a single Carbon atom substituting a Nitrogen atom"
request = "4x4x2 Gallium Nitride supercell, wurtzite phase, with a Gallium vacancy and an adjacent Oxygen atom substituting a Nitrogen atom"

workflow = DFTWorkflow(max_refinement_cycles=3, generator_model = "gemini-2.5-pro", validator_model= "gemini-2.5-pro", output_dir='GaN_def')

result = workflow.run_complete_workflow(request)
print(workflow.get_summary(result))
