# thesis_software
Software devolped for the PhD dissertation Causes and consequences of matching habitat choice, an alternative route towards adap-tive evolution

Here are the pyhton scripts referenced in Chapters 2, 4 and 5.

Chapter 2: assortative_mating_simulation_CH2.py
------------------------------------------------
This script simulates the degree of assortative mating expected from the two phenotypes of flies (white and wild type).
The flies were tested separatedly and this script assumes they segregate in the same way when released together. 

Chapter 4: assortative_mating_simulation_CH4.py
------------------------------------------------
This script simulates the results of assortative mating expected for different strengths of habitat-phenotype covariation.
This data is then used to assess the robustness of the observed empirical results.

Chapter 5: adaptation_tools.py
------------------------------
This script contains the code needed to run the individual-based simulation model described in chapter 5. 
It defines the Population class and its method .advance_generation()
It also includes all the required functions and extra functions to generate graphic output. 
