# Complex Systems: Polarisation in Networks - Team 4 (networked)

## Project Description
Political polarization has become an increasingly higher problem, reaching an all time high in 50 years in the united states (Jones et al. (2019)). Explaining the emergence of polarization in society becomes difficult as other studies show that many people avoid revealing their own personal politcal ideology amongst others (Klar et al. (2018)). This project aims to explain this paradox using network models based upon Tokita et al. (2021), who examine media-driven political polarization as a social process and propose a mechanism for its emergence. Using their framework, this project explores how information cascades from different news sources can lead individuals to self-sort their links under threshold based rules. These rules are driven by news content rather than political preferences, with content either received directly from news outlets or shared by peers in social networks. Tokita et al. make a key distinction: while people assess direct news based on alignment with their political preferences, they often overlook the source when evaluating news reshared by peers. This oversight can influence individuals to reorganize their social connections when comparing content from preferred sources with peer-shared news. The model demonstrates how polarization can arise through simple link-breaking and link-forming rules, leading to biased cascades in politically homogeneous neighbourhoods. As an extension, this project examines how different initial network structures affect polarization by comparing Barabási-Albert (scale-free) and Erdős-Rényi (random) networks, as many social networks tend to follow a power-law distribution (Muchnik et al. (2013)), building on the research of Wang et al. (2024)

## Model details
The model consists of nodes which represent a political oriented individual (either "Left" or "Right") connected via bidirectional edges. They are randomly linked to other individuals, maintaining approximately the same number of connections per individual. Each individual contains a certain 'believe' threshold, meaning how easily they become 'convinced' by the news: This is called an activated state. At each time step, a minority of individuals receive news from their source, which is annotated with an importance value (derived from the covariance among distinct news sources), influencing whether an individual’s  activation threshold is met. If a sufficient fraction of an individual's neighbors activate, they also become activated, triggering an information cascade. Once activated, an individual evaluates whether they would have become activated by receiving the same news directly from their source. If not, it breaks a link with an active node and a new link between two random nodes is formed, preserving the network size. This link reorganization, governed by news content rather than political labels, forms the basis of the emergence of polarized clusters. 
#### An animation can be seen below:


<div align="center">
    <img src="animations/algorithm.gif" alt="Visualisation GIF" width="800"/>
</div>


#### The algorithm consists of multiple update rounds (***Iterations***). 

### In each update round:

#### 1. News Sampling
- Left and Right news is brought out with a certain ***correlation*** value:
highly correlated news means a value of 1 and the opposite means a value of -1. 
- A fraction of individuals become selected and turn into **<span style="color:orange">samplers</span>** if the media importance exceeds their threshold.

#### 2. Information Cascades
- Inactive individuals become **<span style="color:lightgreen">activated</span>** if the fraction of activated neighbors exceeds their threshold.
- This process (Information cascades) iterates until equilibrium is reached and no new individuals become activated.

#### 3. Network Readjustment
- A random active node that has been activated against its own political news preference (i.e., the politically oriented news value is below its threshold) will **<span style="color:red">break</span>** a random connection with an active neighbor.
- If a connection has been broken, a new random connection is **<span style="color:green"> formed </span>** (***Alteration***).


### Two types of networks are implemented:
#### 1. Random Network
- This network randomly connects nodes with degree **k** using probability **p**. 

#### 1. Scale-Free network
- This network connects starts with **m** nodes and connects random nodes using the proportional to its degree. 
- Each node will have atleast **m** connections and this property is mainted during the network adjustment process, to keep sacle-free properties throughout the simulation.

## Experimentation
#### For a range of correlation values (-1.0, -0.8, -0.6, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0), three metrics can be measured to determine polarization within the network:
#### 1. Cascade Sizes
A cascade forms when activated nodes sequentially trigger their neighbors, and cascades merge if they share one or more common nodes. The polarization of a cascade (how imbalanced the proportion of political identities is within it) serves as a metric for overall network polarization.

#### 2. Assortativity
The assortativity coefficient describes the tendency for a node to connect with another node with the same characteristics. In this context, that characteristic is political identity. If the coefficient is 0, a node with political identity L has the same amount of L and R connections on average. If the coefficient is greater than 0, a node with identity L has more L connections on average. With this in mind, we can you this coefficient as a measure for polarization. 

#### 3. Social Ties
A social tie is measured by determining if connections between nodes are lost/gained depending on the political identity, before and after the simulation.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/salomepoulain/complex_systems_networked.git
   cd complex_systems_networked
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

## Usage
- Open `main.ipynb` in Jupyter Notebook.
- Follow the steps outlined in the notebook to run simulations and visualize network behaviors.

## Project Structure
```
complex_systems_networked/
│-- main.ipynb                     # Main Jupyter Notebook to run everything
│-- requirements.txt               # Dependencies
│-- README.md                      # Project documentation
│-- presentation_group4.pptx       # Poweropint presentation
│
│-- animations/                    # Contains GIFs and visualizations
│   │-- cascade_distribution/
│       │-- random/
│       │-- scale_free/
│
│-- networks/                      # Final State Network data files 
│   │-- random_2/
│   │-- scale_free/
│   │-- scale_free_2/
│
│-- plots                          # Plots of experiment results    
│   │-- experiment_results/
│       │-- cascade_distribution/
│           │-- both/
│           │-- random/
│           │-- scale_free/
│-- src/                           # Source code directory
│   │-- classes/
│   │   │-- network.py
│   │   │-- node.py
│   │-- assortativity_exp.py
│   │-- experimentation.py
│   │-- social_ties_exp.py
│   │-- visualization.py
│
│-- statistics/                    # Statistic significance data
│   │-- assortativity/
│   │-- cascades/
```
`Networks/` contains folders 

## Dependencies
All dependencies are listed in `requirements.txt`. Ensure you install them before running the project.

## References

Muchnik, L., Pei, S., Parra, L. C., Reis, S. D. S., Andrade Jr., J. S., Havlin, S., & Makse, H. A. (2013). Origins of power-law degree distribution in the heterogeneity of human activity in social networks. Scientific Reports, 3, 1783. https://doi.org/10.1038/srep01783

Tokita, C. K., Guess, A. M., & Tarnita, C. E. (2021). Polarized information ecosystems can reorganize social networks via information cascades. Proceedings of the National Academy of Sciences, 118(50), e2102147118. https://doi.org/10.1073/pnas.2102147118

Wang, H., Li, Y., & Chen, J. (2024). Three-stage cascade information attenuation for opinion dynamics in social networks. Entropy, 26(851). https://doi.org/10.3390/e26100851

S. Klar, Y. Krupnikov, J. B. Ryan, Affective polarization or partisan disdain? Untangling
a dislike for the opposing party from a dislike of partisanship. Public Opin. Q. 82,
379–390 (2018).

J. M. Jones, Americans continue to embrace political independence. Gallup,
7 January 2019. https://news.gallup.com/poll/245801/americans-continue-embrace-
political-independence.aspx. Accessed 2 February 2025

## Authors
- Chris Hoynck van Papendrecht
- Job Marcelis
- Salomé Poulain
- Thomas Nijkamp

## License
This project is licensed under the terms specified in the `LICENSE` file.