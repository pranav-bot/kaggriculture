class Actions:
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

    PASS = "PASS"

    WATER = "WATER"
    HARVEST = "HARVEST"
    FERTILIZE = "FERTILIZE"
    DIG = "DIG"

    DROP = "DROP"

    @staticmethod
    def plant(crop_name: str) -> list:
        return ["PLANT", crop_name.upper()]

    @staticmethod
    def place(item_name: str) -> list:
        return ["PLACE", item_name.upper()]

    @staticmethod
    def pickup(quantity: int) -> list:
        return ["PICKUP", quantity]

    @staticmethod
    def buy_land() -> list:
        return ["BUY_LAND"]

    @staticmethod
    def hire() -> list:
        return ["HIRE"]

    @staticmethod
    def buy_seed(item_name: str, quantity: int):
        return ["BUY_SEED", item_name.upper(), quantity]

    @staticmethod
    def buy_animal(animal_name: str, quantity: int) -> list:
        return ["BUY_ANIMAL", animal_name.upper(), quantity]

    @staticmethod
    def buy_product(item_name: str, quantity: int) -> list:
        return ["BUY_ANIMAL", item_name.upper(), quantity]

    @staticmethod
    def sell(item_name: str, quantity: int) -> list:
        return ["SELL", item_name.upper(), quantity]