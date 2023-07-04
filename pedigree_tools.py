import os
import pandas as pd
import numpy as np
import msprime
import networkx as nx
import tskit
import matplotlib.pyplot as plt
import itertools

def show_plots():
    plt.ioff()
    plt.show()

class Pedigree(tskit.TableCollection):
    '''
    A wrapper for a tskit TableCollection. Should be usable anywhere a TableCollection is usable.
    '''

    def __init__(self, table_collection: tskit.TableCollection):
        table_copy = table_collection.copy()
        self.__dict__ = table_copy.__dict__
        self.generate_additional_pedigree_data()
        return
    
    def generate_additional_pedigree_data(self):
        self.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
        self.children_table = self.generate_children_table()
        self.proband_individuals = self.get_proband_individuals()
        self.proband_descendent_table = self.generate_proband_descendent_table()
        return
    
    def export_simple(self, filename):
        return export_simple(self, filename)
    
    def generate_proband_descendent_table(self):
        return generate_proband_descendent_table(self)
    
    def get_proband_genealogy(self, proband_individuals: list):
        return get_proband_genealogy(self, proband_individuals)

    def get_descendents_genealogy(self, ancestral_nodes: list):
        return get_descendents_genealogy(self, ancestral_nodes)

    def predicted_contribution(self):
        return predicted_contribution(self)
    
    def predict_contributions_coalescences(self):
        return predict_contributions_coalescences(self)

    def get_proband_individuals(self):
        return get_proband_individuals(self)
    
    def generate_children_table(self):
        return generate_children_table(self)

    def display(self, labels:dict=None, font_size=12, block=True):
        draw_pedigree(self, labels=labels, font_size=font_size, block=block)

    def relabel(self):
        relabel(self)

def to_pedigree(pedigree: tskit.TableCollection) -> Pedigree:
    '''
    Converts a tskit TableCollection to a pedigree_tools Pedigree, in-place.
    It is recommended to use Pedigree() instead of this method,
    as Pedigree() returns a deep copy of the input TableCollection
    '''
    pedigree.__class__ = Pedigree
    pedigree.generate_additional_pedigree_data()
    return pedigree


def relabel(pedigree: tskit.TableCollection):
    '''
    Sets 'label' metadata to the current individual ids. Useful for sorting/trimming pedigrees and retaining the same individual labels.
    '''
    for individual_id, individual in enumerate(pedigree.individuals):
        pedigree.individuals[individual_id] = individual.replace(metadata={'label':individual_id})
    return


def generate_children_table(pedigree: tskit.TableCollection):
    '''
    Generates a dictionary mapping each individual to its children.
    '''
    children_table = {}
    for individual_id, individual in enumerate(pedigree.individuals):
        for parent in individual.parents:
            if parent != -1:
                if parent not in children_table:
                    children_table[parent] = set()
                children_table[parent].add(individual_id)
    return children_table

def generate_proband_descendent_table(pedigree: tskit.TableCollection):
    '''
    Generates a dictionary mapping each individual to its proband descendents.
    '''
    proband_descendent_table = {}
    if isinstance(pedigree, Pedigree):
        proband_individuals = pedigree.proband_individuals
        children_table = pedigree.children_table
    else:
        proband_individuals = get_proband_individuals(pedigree)
        children_table = generate_children_table(pedigree)
    for individual_id in proband_individuals:
        proband_descendent_table[individual_id] = {individual_id}
    
    def _get_proband_descendents(individual_id):
        if individual_id not in proband_descendent_table:
            proband_descendent_table[individual_id] = set()
            if individual_id not in children_table:
                return proband_descendent_table[individual_id]
            for child_id in children_table[individual_id]:
                proband_descendent_table[individual_id].update(_get_proband_descendents(child_id))

        return proband_descendent_table[individual_id]

    for individual_id in range(len(pedigree.individuals)):
        _get_proband_descendents(individual_id)
    return proband_descendent_table

def generate_pedigree(num_probands=None, population_size=None, end_time=None, num_pedigrees=None, random_seed=None) -> Pedigree | list[Pedigree]:
    '''
    Generates a pedigree with the specified number of sample individuals, out of a population of the specified size, with the specified number of generations.
    '''
    if population_size is None:
        raise ValueError('Population size must be specified.')
    if num_probands is not None and num_probands > population_size:
        raise ValueError('Number of probands cannot be larger than the size of the population.')
    
    pedigrees = []
    for i in range(1 if num_pedigrees == None else num_pedigrees):
        pedigree = msprime.pedigrees.sim_pedigree(population_size=population_size, random_seed=random_seed+i, sequence_length=1, end_time=end_time)
        to_pedigree(pedigree)
        if num_probands is None:
            pedigrees.append(pedigree)
        else:
            pedigrees.append(pedigree.get_proband_genealogy(itertools.islice(pedigree.proband_individuals, num_probands)))

    if num_pedigrees == None:
        return pedigrees[0]
    else:
        return pedigrees

def get_proband_individuals(pedigree: tskit.TableCollection):
    ped_ts = pedigree.tree_sequence()
    sample_ids = set()
    for node_id in ped_ts.samples():
        sample_ids.add(ped_ts.nodes_individual[node_id])
    return sample_ids

def get_proband_genealogy(pedigree: tskit.TableCollection, chosen_probands):
    '''
    Gets the genealogical history of the chosen probands from the pedigree.
    In other words, this removes all individuals from a pedigree that are not ancestral to the chosen probands.
    Input: A number of sample individuals and a pedigree including those individuals
    Output: A pedigree with all individuals other than the chosen probands and their ancestors removed.
    '''
    if isinstance(chosen_probands, set):
        chosen_proband_set = chosen_probands
    else:
        chosen_proband_set = set(chosen_probands)
    if isinstance(pedigree, Pedigree):
        proband_descendent_table = pedigree.proband_descendent_table
    else:
        proband_descendent_table = generate_proband_descendent_table(pedigree)
    relevant_nodes = [node_id for node_id, node in enumerate(pedigree.nodes) if intersects(chosen_proband_set, proband_descendent_table[node.individual])]
    #print(relevant_nodes)
    subset_pedigree = pedigree.copy()
    subset_pedigree.subset(relevant_nodes)
    to_pedigree(subset_pedigree)
    return subset_pedigree

def individual_stat_from_nodes(node_stats, pedigree: tskit.TableCollection):
    pedigree.sort_individuals()
    #print(node_stats)
    individual_stats = np.zeros(int(len(node_stats)/2))
    #print(individual_stats)
    for node_id, stat in enumerate(node_stats):
        individual_stats[pedigree.nodes[node_id].individual] += stat[0]
    print
    return individual_stats

def intersects(a: set, b: set):
    '''
    Returns True if there is any intersection between a and b
    '''
    return not a.isdisjoint(b)

def get_descendents_genealogy(pedigree: tskit.TableCollection, ancestral_individuals: list):
    '''
    Gets the genealogical history of all descendents of the chosen individuals.
    In other words, this removes all individuals from a pedigree that are not ancestral to the descendents of the chosen node.
    Input: A number of sample individuals and a pedigree including those individuals.
    Output: A pedigree with all individuals other than the descendents of chosen individuals, and their ancestors, removed.
    '''
    chosen_proband_set = set()
    if isinstance(pedigree, Pedigree):
        proband_descendent_table = pedigree.proband_descendent_table
    else:
        proband_descendent_table = generate_proband_descendent_table(pedigree)
    for ancestral_id in ancestral_individuals:
        chosen_proband_set.update(proband_descendent_table[ancestral_id])

    return get_proband_genealogy(pedigree, chosen_proband_set)

def export_simple(pedigree: tskit.TableCollection, filename):
    with open(filename, 'w') as file:
        for individual_id, individual in enumerate(pedigree.individuals):
            file.write(f'{individual_id} {individual.parents[0]} {individual.parents[1]}\n')

def draw_pedigree(pedigree: tskit.TableCollection, labels = None, font_size=12, block=True):
    '''
    Creates a matplotlib plot of a tskit pedigree.
    ''' 
    ped_ts = pedigree.tree_sequence()
    G = nx.DiGraph()
    if labels is None:
        labels = {}
    for ind in ped_ts.individuals():
        if isinstance(ind.metadata, dict):
            if 'label' in ind.metadata:
                labels[ind.id] = ind.metadata['label']
            elif 'file_id' in ind.metadata:
                labels[ind.id] = ind.metadata['file_id']
            if ind.id not in labels:
                labels[ind.id] = ind.id
        time = ped_ts.node(ind.nodes[0]).time
        pop = ped_ts.node(ind.nodes[0]).population
        G.add_node(ind.id, time=time, population=pop)
        for p in ind.parents:
            if p != tskit.NULL:
                G.add_edge(ind.id, p)
    pos = nx.multipartite_layout(G, subset_key="time", align="horizontal")
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    node_colours = [colours[node_attr["population"]] for node_attr in G.nodes.values()]
    fig, axes = plt.subplots(nrows=1, ncols=1)
    
    nx.draw_networkx(G, pos, with_labels=True, labels=labels, node_color=node_colours, font_size=font_size, ax=axes, bbox={'fc':'r'})
    if block==False:
        plt.ion()
    plt.show()

def predicted_contribution(pedigree: tskit.TableCollection):
    pedigree.sort_individuals()
    ped_ts = pedigree.tree_sequence()
    proband_ids = get_proband_individuals(pedigree)
    contribution = np.zeros(ped_ts.num_individuals)
    #print([ind for ind in ped_ts.individuals()])
    #print(sample_ids)
    for ind in reversed(ped_ts.individuals()):
        if ind.id in proband_ids:
            contribution[ind.id] = 2
        if ind.parents[0] != -1:
            contribution[ind.parents[0]] += 0.5*contribution[ind.id]
            #print(ind.id, contribution[ind.id], ind.parents[0], contribution[ind.parents[0]])
        if ind.parents[1] != -1:
            contribution[ind.parents[1]] += 0.5*contribution[ind.id]
            #print(ind.id, contribution[ind.id], ind.parents[1], contribution[ind.parents[1]])
    return contribution

def predict_contributions_coalescences(pedigree: tskit.TableCollection):
    pedigree.sort_individuals()
    ped_ts = pedigree.tree_sequence()
    proband_ids = get_proband_individuals(pedigree)
    contributions = [[] for i in range(ped_ts.num_individuals)]
    contribution_sum = np.zeros(ped_ts.num_individuals)
    coalescences = np.zeros(ped_ts.num_individuals)
    for ind in reversed(ped_ts.individuals()):
        if ind.id in proband_ids:
            contributions[ind.id] = [2]
        contribution_sum[ind.id] = sum(contributions[ind.id])
        coalescences[ind.id] = (sum(contributions[ind.id])**2 - sum(contribution**2 for contribution in contributions[ind.id]))/4
        if ind.parents[0] != -1:
            contributions[ind.parents[0]].append(0.5*contribution_sum[ind.id])
            #print(ind.id, contribution[ind.id], ind.parents[0], contribution[ind.parents[0]])
        if ind.parents[1] != -1:
            contributions[ind.parents[1]].append(0.5*contribution_sum[ind.id])
            #print(ind.id, contribution[ind.id], ind.parents[1], contribution[ind.parents[1]])
    return contribution_sum, coalescences

def predicted_cross_coalescent(pedigree: tskit.TableCollection):
    contribution = predicted_contribution(pedigree)
    return contribution

def load_and_verify_pedigree(fname):
    """
    Output: verified four column pedigree dataframe

    Input:
    fname: string giving location of txt_ped-formatted genealogy.
    columns represent:  ind, mother, father, generation (lon, lat are optional)

    This function:
    checks if file exists,
    sorts table in decending genealogical order (oldest to newest)
    identify pedigree founders and assign -1 values
    """
    # ensure file exists
    try:
        f = open(fname, 'rb')
    except FileNotFoundError:
        print("file {} does not exist".format(fname))
        raise

    # genealogy_table instead of fp

    #  load the genealogy file
    fp = pd.read_csv(fname)
    # reverse sort by pseudo-time such that parents are before their children
    #fp = fp.sort_values(["generation"], ascending = (False)).reset_index(drop=True)

    # identify and recode founder individuals

    # these are the individuals in the pedigree
    ped_inds = fp["ind"].values
    # assign -1 to founding fathers
    fp.loc[~fp["father"].isin(ped_inds), "father"] = -1
    # assign -1 to founding mothers
    fp.loc[~fp["mother"].isin(ped_inds), "mother"] = -1

    return fp

def convert_txt_to_msprime(inFile: str, outFile: str):
    """
    Converts a txt_ped-formatted genealogy to the msprime format, and saves it as new file.
    :param str inFile: Name of the input text file.
    :param str inFile: Name of the output text file.
    """
    fp = load_and_verify_pedigree(inFile)
    fp.rename(columns = {'ind':'id', 'mother':'parent0', 'father':'parent1', 'generation':'time' }, inplace = True)

    fp = fp.replace(-1, '.')
    fp["time"] = fp["time"].astype(float)
    fp["is_sample"] = 1
    #print(fp.to_string())
    prep = open(outFile, 'w')
    prep.write('# ')
    prep.close()
    fp.to_csv(outFile, sep=' ', mode='a', index=False)

def add_individuals_to_pedigree(pb, text_pedigree, f_pop, p_pop):
    """
    Output: PedigreeBuilder object built from text_pedigree

    Input:
    pb: an msprime builder pedigree with a predefined demography
    text_pedigree: four column text pedigree from load_and_verify_genealogy

    This function:
    loops through each individual in text pedigree
    adds individual to msprime pedigree with:
    parents, time, population and metadata of individual_name from text pedigree
    """
    # dictionaries linking text_pedigree ids to msprime ids
    txt_ped_to_tskit_key = {}

    # determine if lon lat present in text pedigree
    if {'lon', 'lat'}.issubset(text_pedigree.columns):
        geo = True
    else :
        geo = False

    # determine if marriage date present in text pedigree
    if {'datem'}.issubset(text_pedigree.columns):
        date = True
    else :
        date = False

    # determine if marriage decade present in text pedigree
    if {'decade'}.issubset(text_pedigree.columns):
        decade = True
    else :
        decade = False

    # determine if new id is present in text pedigree
    if {'new_id'}.issubset(text_pedigree.columns):
        new_id = True
    else :
        new_id = False

    # for each individual in the genealogy
    for i in text_pedigree.index:
        # relevant information to load into PedigreeBuilder
        ind_time = text_pedigree["generation"][i]
        ind_id = text_pedigree["ind"][i]
        father_id = text_pedigree["father"][i]
        mother_id = text_pedigree["mother"][i]

        # add father
        if father_id == -1 :
            father = pb.add_individual(time=ind_time+1,
                                       population=f_pop,
                                       metadata={"individual_name": str(father_id)})
        else:
            try:
                father = txt_ped_to_tskit_key[father_id]
            except KeyError:
                print("father key missing, check order of dictionary construction")
                raise

        # add mother
        if mother_id == -1 :
            mother = pb.add_individual(time=ind_time+1,
                                       population=f_pop,
                                       metadata={"individual_name": str(mother_id)})

        else:
            try:
                mother = txt_ped_to_tskit_key[mother_id]
            except KeyError:
                print("mother key missing, check order of dictionary construction")
                raise

        if geo and date and decade and new_id :
            metadata={"individual_name": str(ind_id),
                      "geo_coord":[text_pedigree["lat"][i],text_pedigree["lon"][i]],
                      "date":str(text_pedigree["datem"][i]),
                      "decade":str(text_pedigree["decade"][i]),
                      "new_id":str(text_pedigree["new_id"][i]),
                      }
        elif geo and date and new_id :
            metadata={"individual_name": str(ind_id),
                      "geo_coord":[text_pedigree["lat"][i],text_pedigree["lon"][i]],
                      "date":str(text_pedigree["datem"][i]),
                      "new_id":str(text_pedigree["new_id"][i]),
                      }
        elif geo and decade and new_id :
            metadata={"individual_name": str(ind_id),
                      "geo_coord":[text_pedigree["lat"][i],text_pedigree["lon"][i]],
                      "decade":str(text_pedigree["decade"][i]),
                      "new_id":str(text_pedigree["new_id"][i]),
                      }
        elif new_id :
            metadata={"individual_name": str(ind_id),
                      "new_id":str(text_pedigree["new_id"][i]),
                      }
        else :
            metadata={"individual_name": str(ind_id)}
        # add individual
        child = pb.add_individual(time=ind_time,
                                  parents=[mother,father],
                                  population=p_pop,
                                  metadata=metadata)

        # update dictionary for downstream
        txt_ped_to_tskit_key[ind_id] = child # store for later use (?)

    return pb

def del_sensitive_metadata(md):
    del md["date"]
    del md["new_id"]

    return md

def censor_pedigree(ts):
    """
    Output: a censored tree sequence (i.e. without parent-child links or IDs)

    Input:
    ts: a tree sequence

    This function:
    removes all sensitive metadata from the input text pedigree
    specifically, it removes:
    - individual_names
    - parents of each individual
    """

    tables = ts.dump_tables()

    new_metadata = [del_sensitive_metadata(i.metadata) for i in tables.individuals]

    validated_metadata = [
        tables.individuals.metadata_schema.validate_and_encode_row(row) for row in new_metadata
    ]
    tables.individuals.packset_metadata(validated_metadata)

    # remove parents
    tables.individuals.packset_parents([[]] * tables.individuals.num_rows)

    censored_ts = tables.tree_sequence()

    return(censored_ts)

def clean_pedigree_for_publication(ts):
    """
    Output: a cleaned tree sequence file (i.e. clean metadata, provenances, etc.)

    Input:
    ts: a tree sequence

    This function:
    removes useless metadata and provenances from the input tree sequence
    specifically:
    - sets geographical coordinates to `location`
    - only keeps the first two provenance entries
    - only keeps metadata cleared for publication
    """
    # ensure pedigree is censored
    ts = censor_pedigree(ts)

    # load tables
    tables = ts.dump_tables()
    # only keep first two entries of provenances
    tables.provenances.truncate(2)
    # get the lat and lon for each individual
    location = np.array(list(ind.metadata["geo_coord"] for ind in ts.individuals()))

    n = ts.num_individuals

    # set the location to lat/lon
    tables.individuals.set_columns(
            flags=tables.individuals.flags,
            location=location.reshape(2 * n),
            location_offset=2 * np.arange(n + 1, dtype=np.uint64),
            metadata=tables.individuals.metadata,
            metadata_offset=tables.individuals.metadata_offset)

    clean_ts = tables.tree_sequence()
    return(clean_ts)

def simulate_genomes_with_known_pedigree(
                                         text_pedigree,
                                         demography,
                                         model = "hudson",        # model to recapitulate tree
                                         f_pop = "EUR",           # population id of founders
                                         p_pop = "EUR",           # population id in pedigree
                                         mutation_rate = 3.62e-8,
                                         rate_map = 1.20e-8,
                                         sequence_length = 1,
                                         sequence_length_from_assembly = 1,
                                         centromere_intervals = [0,0],
                                         censor = True,
                                         seed = 123
                                         ):
    """
    Output: simulated genomes using input text pedigree

    Input:
    text_pedigree: four column text pedigree from load_and_verify_genealogy
    demography: msprime demography specification
    model: used to recapitulate the fixed pedigree -- "hudson" or "WF"
    f_pop: population id of founders
    p_pop: population id in pedigree
    sequence_length: genome length of tree sequence
    sequence_length_from_assembly: length including telomeres
    rate_map: recombination rate map defined by load_rate_map
    mtuation_rate: mutation rate used for dropping mutations down tree sequence
    seed: random seed used in simulations

    This function:
    initializes an msprime PedigreeBuilder from demography
    builds a pedigree using the input text_pedigree
    runs msprime.sim_ancestry within fixed pedigree (default diploid)
    using the recombination rate provided
    drops mutations down tree using provided mutation rate
    """
    # demography used to recapitulate beyond input pedigree
    pb = msprime.PedigreeBuilder(demography)

    # build pedigree using input pedigree
    pb = add_individuals_to_pedigree(pb, text_pedigree, f_pop, p_pop)

    # check simple model https://github.com/tskit-dev/msprime/blob/57ef4ee3267cd9b8e711787539007b0cde94c55c/tests/test_pedigree.py#L151

    # initial state of tree sequence
    ts = pb.finalise(sequence_length = sequence_length)

    # simulation within fixed pedigree
    ts = msprime.sim_ancestry(
        initial_state = ts,
        recombination_rate = rate_map,
        model = "fixed_pedigree",
        random_seed = seed + 100
        )

    # simulation beyond fixed pedigree
    ts = msprime.sim_ancestry(
        initial_state = ts,
        recombination_rate = rate_map,
        demography = demography,
        random_seed = seed + 200,
        model = model # Could also do WF
        )
    # drop mutations down the tree
    ts = msprime.sim_mutations(
              ts,
              rate = mutation_rate,
              random_seed = seed + 300
              )

    if(censor): ts = censor_pedigree(ts)

    # remove centromere
    ts = ts.delete_intervals(intervals = centromere_intervals)
    # modify sequence length to include `right` telomere
    tables = ts.dump_tables()
    tables.sequence_length = sequence_length_from_assembly
    ts = tables.tree_sequence()
    return ts

def simulation_sanity_checks(ts, ped):
    """
    ts is the output of run_fixed_pedigree_simulation
    text_pedigree is the output of load_and_verify_genealogy
    """

    # probands are by definition at generation 0
    probands = ped.loc[ped['generation'] == 0]["ind"].values

    # reacall diploids have two nodes per sample
    assert ts.num_samples == 2 * len(probands)

    # TODO : assert samples IDs are the correctly stored
    #ts.tables.individuals[5].metadata['individual_name']

    pass

def drop_mutations_again(
                         ts,
                         inside_mut = 2.36e-8,
                         outside_mut = 3.62e-8,
                         seed = 0,
                         ):
    """
    Output: tree sequence with a new set of mutations

    Input:
    ts: a tree sequence
    inside_mut: mutation rate used _inside_ fixed pedigree
    outside_mut: mutation rate use _outside_ fixed pedigree

    NOTE: outisde_mut should match the one used in the demographic model.

    This function:
    removes all sites and mutations from the input tree sequence
    drops mutations down tree using provided the two mutation rates
    the optional seed argument can be used to generate new simulations
    """

    # load tables
    tables = ts.dump_tables()
    # remove sites
    tables.sites.clear()
    # remove mutations
    tables.mutations.clear()
    # turn this back into a tree sequence
    ts_nomuts = tables.tree_sequence()

    # cut tree seuquence into two based on start_time and end_time
    # ts_nomuts_inside =
    # ts_nomuts_outside =

    # drop mutations down the tree
    ts_inside = msprime.sim_mutations(
              ts_nomuts_inside,
              rate = inside_mut,
              random_seed = seed
              )

    # drop mutations down the tree
    ts_outside = msprime.sim_mutations(
              ts_nomuts_outside,
              rate = outside_mut,
              random_seed = seed
              )

    # ts_out = ts_inside + ts_outside

    #return(ts)
    pass

