# Author: Zbigniew Dynowski
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from typing import List, Optional, Dict, Any
from random import choice, choices, random

import config as cfg

from algorithm.edt_components import Node, CandidateTree, NodeType
from algorithm.utils import evaluate_candidates, choose_node_split_params


class EvolutionaryDecisionTree:
    def __init__(
            self,
            data_info: Dict[str, Dict[str, Any]],
            all_labels: np.ndarray,
            population_size: int = cfg.POPULATION_SIZE,
            split_prob: float = cfg.SPLIT_PROBABILITY,
            expected_height: int = cfg.EXPECTED_TREE_HEIGHT,
            mutation_prob: float = cfg.P_MUTATION,
            crossover_prob: float = cfg.P_CROSSOVER
    ):
        self.population_size: int = population_size
        self.tournament_size: int = cfg.TOURNAMENT_SIZE
        self.split_prob: float = split_prob
        self.expected_height: int = expected_height
        self.mutation_prob: float = mutation_prob
        self.crossover_prob: float = crossover_prob
        self.max_iters: int = cfg.MAX_ITERATIONS
        self.stall_iters: int = cfg.STALL_ITERATIONS
        self.tree: Optional[CandidateTree] = None
        self.data_info: Dict[str, Dict[str, Any]] = data_info
        self.all_labels: np.ndarray = all_labels
        print(f'p_size:{self.population_size}, split_p:{self.split_prob}, e_height:{self.expected_height}, m_p:{self.mutation_prob}, c_p:{self.crossover_prob}')

    def fit(self, data: pd.DataFrame, labels: pd.Series) -> None:
        population: List[CandidateTree] = [
            CandidateTree.generate_tree(self.split_prob, data, labels) for _ in range(self.population_size)
        ]
        evaluate_candidates(population, data, labels)
        best_tree: CandidateTree = population[0]
        stall_iterations: int = 0

        for _ in range(self.max_iters):
            if stall_iterations >= self.stall_iters:
                break

            new_population: List[CandidateTree] = []
            selected_individuals: List[CandidateTree] = self.select(population)
            for individual in selected_individuals:
                individual = individual.copy()
                if random() < self.crossover_prob:
                    individual = self.crossover(selected_individuals, individual)
                if random() < self.mutation_prob:
                    individual = self.mutate(individual)
                new_population.append(individual)

            evaluate_candidates(new_population, data, labels)
            population = self.succession(population, new_population)
            best_candidate: CandidateTree = min(population, key=lambda x: x.fitness)
            if best_candidate.fitness < best_tree.fitness:
                best_tree = best_candidate
                stall_iterations = 0
            else:
                stall_iterations += 1

        self.tree = best_tree

    def select(self, population: List[CandidateTree]) -> List[CandidateTree]:
        elite: List[CandidateTree] = sorted(population, key=lambda x: x.fitness)[:cfg.ELITE_SIZE]
        selected_individuals: List[CandidateTree] = elite

        for _ in range(self.population_size - cfg.ELITE_SIZE):
            rank = choices(population, k=self.tournament_size)
            selected_individuals.append(min(rank, key=lambda x: x.fitness))

        return [individual.copy() for individual in selected_individuals]

    def crossover(self, all_selected: List[CandidateTree], first_parent: CandidateTree) -> CandidateTree:
        second_parent: CandidateTree = choice(all_selected)
        first_random_node: Node = first_parent.root.random_node()
        second_random_node: Node = second_parent.root.random_node().copy()

        if first_random_node.parent is not None:
            parent = first_random_node.parent
            second_random_node.parent = parent
            if first_random_node is parent.left_child:
                parent.left_child = second_random_node
            elif first_random_node is parent.right_child:
                parent.right_child = second_random_node
            else:
                Exception('Chosen node is not child of its parent')
            return first_parent
        else:
            second_random_node.parent = None
            return CandidateTree(second_random_node)

    def mutate(self, individual: CandidateTree) -> CandidateTree:
        random_node: Node = individual.root.random_node()
        if random_node.node_type == NodeType.LEAF:
            random_node.label = choice(self.all_labels)
        else:
            attribute, attribute_idx, threshold = choose_node_split_params(self.data_info)
            random_node.attribute = attribute
            random_node.attribute_idx = attribute_idx
            random_node.threshold = threshold
        return individual

    def succession(self, population: List[CandidateTree], new_population: List[CandidateTree]) -> List[CandidateTree]:
        return sorted(population + new_population, key=lambda x: x.fitness)[:self.population_size]

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.tree is None:
            raise Exception('fit() must be called before predict!')
        return self.tree.predict(data)

    def score(self, data: pd.DataFrame, true_labels: pd.Series) -> float:
        predicted_labels: np.ndarray = self.predict(data)
        return accuracy_score(true_labels, predicted_labels)
