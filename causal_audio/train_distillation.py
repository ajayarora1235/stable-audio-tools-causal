class DiffusionDistillationTrainer:
    def __init__(self, model, scheduler, num_train_timestep, device):
        self.model = model
        self.scheduler = scheduler
        self.num_train_timestep = num_train_timestep
        self.device = device

        if config.distillation_loss == "dmd":
            self.distillation_model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distillation loss type")

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        dataset = ODERegressionLMDBDataset(
                config.data_path, max_pair=int(1e8))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler)
        self.dataloader = cycle(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.distillation_model.generator)
        critic_state_dict = fsdp_state_dict(
            self.distillation_model.fake_score)
        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict
        }

        if self.is_main_process:
            save_path = os.path.join(
                config.output_dir, f"distillation_model_{self.step}.pt")
            torch.save(state_dict, save_path)
            print(f"Model saved to {save_path}")

    def training_step(self, batch, batch_idx):
        self.distillation_model.eval()
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
        VISUALIZE = self.step % self.config.log_iters == 0 and not self.config.no_visualize

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        if not self.backward_simulation:
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype)
        else:
            text_prompts = next(self.dataloader)
            clean_latent = None

        batch_size = len(text_prompts)
        audio_shape = list(self.config.audio_shape)
        audio_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.distillation_model.text_encoder(
                text_prompts=text_prompts)

            conditioning = self.distillation_model.conditioner(metadata, self.device)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.distillation_model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        if TRAIN_GENERATOR:
            generator_loss, generator_log_dict = self.distillation_model.generator_loss(
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent
            )

            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_grad_norm = self.distillation_model.generator.clip_grad_norm_(
                self.max_grad_norm)
            self.generator_optimizer.step()
        else:
            generator_log_dict = {}

        # Step 4: Train the critic
        critic_loss, critic_log_dict = self.distillation_model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self.distillation_model.fake_score.clip_grad_norm_(
            self.max_grad_norm)
        self.critic_optimizer.step()
        

