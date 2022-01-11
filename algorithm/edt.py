import numpy as np

from time import time
from typing import List, Optional, Dict, Any
from sklearn.metrics import accuracy_score
from random import choice, choices, random, uniform

import config as cfg

from .edt_components import Node, CandidateTree, NodeType


class EvolutionaryDecisionTree:
    def __init__(self):
        self.population_size: int = cfg.POPULATION_SIZE
        self.tournament_size: int = cfg.TOURNAMENT_SIZE
        self.split_prob: float = cfg.SPLIT_PROBABILITY
        self.expected_height: int = cfg.EXPECTED_TREE_HEIGHT
        self.mutation_prob: float = cfg.P_MUTATION
        self.crossover_prob: float = cfg.P_CROSSOVER
        self.max_iters: int = cfg.MAX_ITERATIONS
        self.stall_iters: int = cfg.STALL_ITERATIONS
        self.tree: Optional[CandidateTree] = None

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        start = time()
        population: List[CandidateTree] = [
            CandidateTree.generate_tree(self.split_prob, data, labels) for _ in range(self.population_size)
        ]
        self.evaluate_population(population, data, labels)

        best_tree: CandidateTree = population[0]
        stall_iterations: int = 0

        for _ in range(self.max_iters):
            if stall_iterations >= self.stall_iters:
                break

            new_population: List[CandidateTree] = []
            selected_individuals: List[CandidateTree] = self.select(population)

            for j, individual in enumerate(selected_individuals):
                if random() < self.crossover_prob:
                    individual = self.crossover(selected_individuals, individual)
                if random() < self.mutation_prob:
                    individual = self.mutate(individual)
                new_population.append(individual)

            self.evaluate_population(new_population, data, labels)
            population = self.succession(population, new_population)

            best_candidate: CandidateTree = min(population, key=lambda x: x.fitness)
            if best_candidate.fitness < best_tree.fitness:
                best_tree = best_candidate
                stall_iterations = 0
            else:
                stall_iterations += 1

        self.tree = best_tree
        print(f'processing took: {time() - start}')

    def select(self, population: List[CandidateTree]) -> List[CandidateTree]:
        elite: List[CandidateTree] = sorted(population, key=lambda x: x.fitness)[:cfg.ELITE_SIZE]
        selected_individuals: List[CandidateTree] = elite

        for _ in range(self.population_size - cfg.ELITE_SIZE):
            rank = choices(population, k=self.tournament_size)
            selected_individuals.append(min(rank, key=lambda x: x.fitness))

        return selected_individuals

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
            random_node.label = choice(self.data_classes)
        else:
            random_node.attribute = choice((list(self.data_info.keys())))
            attr_data = self.data_info[random_node.attribute]
            threshold = None
            if attr_data['is_string']:
                threshold = choice(attr_data['possible_values'])
            else:
                threshold = round(uniform(attr_data['min_value'], attr_data['max_value']), 3)
            random_node.threshold = threshold
        return individual

    def evaluate_population(self, population: List[CandidateTree], data: np.ndarray, labels: np.ndarray) -> None:
        for individual in population:
            error: float = cfg.ALPHA * (1 - individual.score(data, labels))
            height_penalty: float = cfg.BETA * abs(individual.height() - cfg.EXPECTED_TREE_HEIGHT)
            individual.fitness = error + height_penalty

    def succession(self, previous_population: List[CandidateTree],
                   new_population: List[CandidateTree]) -> List[CandidateTree]:
        return sorted(previous_population + new_population, key=lambda x: x.fitness)[:self.population_size]

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise Exception('fit() must be called before predict!')
        return self.tree.predict(data)

    def score(self, data: np.ndarray, true_labels: np.ndarray) -> float:
        predicted_labels: np.ndarray = self.predict(data)
        score = accuracy_score(true_labels, predicted_labels)
        return score
