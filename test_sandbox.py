import scilink

scilink.configure('google', 'AIzaSyB9lyiu0D9gZfHyAfKPIFkcU3a7sa9V-kE')
scilink.configure('futurehouse', '+MZqvbTtjHVywIJ1GWJ8Zw.platformv01.eyJqdGkiOiI1MDZiZjI2OS0wNThmLTRjNDUtYmM1OC1iMDE2NjYyYTBjMGUiLCJzdWIiOiJuaUt3MDBwVk1nUmV4MDhocUg3RTBTRFVXQ3UyIiwiaWF0IjoxNzQ0NzM4OTA5fQ.9xtT+1ZfVaKWHQurUAV69viXqaTh7YSH9nmDZ0DjnQU')
scilink.configure('materials_project', 'M6DAhhJ4OkKldg9RzGWPtowFR1BkZH9C')


from scilink.workflows.dft_workflow import DFTWorkflow

#request = "A heterostructure consisting of a 3-layer slab of MoS2 (2H phase) on top of a 3-layer slab of WS2 (2H phase), with a 3x3 lateral cell. For electronic structure calculations"
#request = "3x3 Cu(100) slab, 4 layers, with a single NH3 molecule adsorbed on hollow site"
#request = "3x3 Ni(111) slab, 4 layers, with CO adsorbed at top site"
#request = "6x6 graphene monolayer, with a double vacancy passivated with hydrogen atoms"
request = "Grain boundary in yttrium barium copper oxide"
#request = "2x2 LaAlO₃ on 2x2 SrTiO₃ interface slab, LaO/TiO₂ termination, with an Oxygen vacancy at the interface plane."

workflow = DFTWorkflow()

result = workflow.run_complete_workflow(request)
print(workflow.get_summary(result))