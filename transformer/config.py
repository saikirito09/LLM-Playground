import os
from datetime import datetime

class TranslationConfigManager:
    def __init__(self):
        self.settings = {
            "batch_size": 8,
            "training_epochs": 5,
            "learning_rate": 1e-4,
            "sequence_length": 350,
            "embedding_dim": 512,
            "data_corpus": 'opus_books',
            "source_language": "en",
            "target_language": "es",
            "checkpoint_dir": "model_checkpoints",
            "model_prefix": "en_es_transformer_",
            "load_checkpoint": "most_recent",
            "tokenizer_template": "tokenizer_{}.json",
            "log_directory": "logs/en_es_transformer"
        }

    def get_settings(self):
        return self.settings.copy()

    def checkpoint_path(self, epoch_identifier):
        checkpoint_name = f"{self.settings['model_prefix']}{epoch_identifier}.pth"
        return os.path.join(self.settings['data_corpus'], self.settings['checkpoint_dir'], checkpoint_name)

    def find_most_recent_checkpoint(self):
        checkpoint_directory = os.path.join(self.settings['data_corpus'], self.settings['checkpoint_dir'])
        if not os.path.exists(checkpoint_directory):
            return None

        checkpoints = [f for f in os.listdir(checkpoint_directory) if f.startswith(self.settings['model_prefix']) and f.endswith('.pth')]

        if not checkpoints:
            return None

        return os.path.join(checkpoint_directory, max(checkpoints, key=lambda f: os.path.getmtime(os.path.join(checkpoint_directory, f))))

    def create_experiment_name(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.settings['source_language']}_{self.settings['target_language']}_{timestamp}"

config_manager = TranslationConfigManager()
