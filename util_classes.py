

class foodRecord:
    def __init__(self, timestamp, foodName, carbs, protein, fat, fiber):
        self.timestamp = timestamp
        self.foodName = foodName
        self.carbs = carbs
        self.protein = protein
        self.fat = fat
        self.fiber = fiber

    def calculate_glycemic_load(self):
        """
        Returns an estimation of the glycemic load of the food eaten.
        """
        GL = 19.27 + (0.39 * self.carbs) - (0.21 * self.fat) - (0.01 * (self.protein**2)) - (0.01 * (self.fiber**2))
        return GL