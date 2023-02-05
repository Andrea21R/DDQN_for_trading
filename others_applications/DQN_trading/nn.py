import tensorflow as tf


class ANN:

    @classmethod
    def build_model(cls, name: str, input_dim: int, output_dim: int, activation_func: str = 'relu'):

        if name == "classic":
            return cls.build_classic_model(input_dim, output_dim, activation_func)
        else:
            raise KeyError("Model not implemented yet. Sorry :)")

    @staticmethod
    def build_classic_model(input_dim: int, output_dim: int, activation_func: str) -> tf.keras.Sequential:
        """
        Returns a ANN with 3 layers (32 neurons at the first level, 64 at the second one and 128 at the third)

        :return:
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(input_dim=input_dim, units=32, activation=activation_func))
        model.add(tf.keras.layers.Dense(units=64, activation=activation_func))
        model.add(tf.keras.layers.Dense(units=128, activation=activation_func))
        model.add(tf.keras.layers.Dense(units=output_dim, activation='linear'))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())

        return model
