"""
TDAE on Stable Diffusion v1.4  —  Pseudocode
=============================================

Symbols
-------
x0          : clean image,  [1,3,H,W], range [0,1]
delta       : adversarial perturbation, same shape as x0
x_adv       : x0 + delta, clamped to [0,1]
c_o         : original text embedding from text_encoder(prompt), [1,L,768]
c_adv       : adversarially perturbed text embedding
z_adv       : reparameterized latent of x_adv, scaled by scaling_factor
z_mean      : deterministic latent mean (no reparam noise)
z_t         : forward-diffused noisy latent at timestep t
ref         : random reference latent (MSE target for noise prediction)
unet        : SD v1.4 UNet denoiser
vae         : SD v1.4 VAE
scheduler   : DDIM / DPM scheduler
alpha_bar_t : cumulative product of (1-beta) up to step t
k           : embedding PGD frequency (apply every k outer iterations)
eps_img     : L-inf budget on delta  (0.03)
alpha       : outer PGD step size    (0.005)
a           : gradient momentum weight (best a = 0.7 per ablation)
N           : number of outer iterations (100)
M           : number of timesteps sampled per gradient estimate (10)
T           : total diffusion steps (1000)
"""


# ─────────────────────────────────────────────────────────────────────────────
# SUB-ROUTINE 1: embedding_pgd
#
# Purpose:
#   Perturb c_o into c_adv (inner PGD, 20 steps) so that the UNet's noise
#   prediction on z_mean is pulled toward z_t (the noisy latent).
#   This makes the text conditioning adversarially inform the gradient for x_adv.
#
# Inputs:  unet, z_mean, z_t, t, c_o
# Output:  c_adv  (same shape as c_o)
# ─────────────────────────────────────────────────────────────────────────────

FUNCTION embedding_pgd(unet, z_mean, z_t, t, c_o):

    c_adv ← c_o + Uniform(-eps_emb, eps_emb)      # random init inside eps ball (eps_emb = 8/255)
    c_adv ← clamp(c_adv, c_o.min, c_o.max)

    FOR _ in 1 .. 20:
        noise_pred ← unet(z_mean, t, encoder_hidden_states=c_adv)
        loss       ← L1(z_t, noise_pred)           # minimize: make pred align with noisy latent
        grad_c     ← ∂loss/∂c_adv

        c_adv ← c_adv - (2/255) * sign(grad_c)    # gradient descent step (inner PGD)
        c_adv ← clamp(c_adv, c_o - eps_emb, c_o + eps_emb)   # project into eps ball
        c_adv ← clamp(c_adv, c_o.min, c_o.max)    # keep in valid embedding range

    RETURN c_adv


# ─────────────────────────────────────────────────────────────────────────────
# SUB-ROUTINE 2: getAvgGrad
#
# Purpose:
#   Estimate the gradient of delta by averaging over M random timesteps.
#   For each timestep t:
#     1. Encode x_adv stochastically → z_adv  (full reparameterization)
#     2. Forward-diffuse z_adv to z_t using DDPM forward process
#     3. Build a random reference latent ref
#     4. Conditionally replace c_o with c_adv (every k outer steps)
#     5. Predict noise with UNet;  loss = MSE(noise_pred, ref)
#     6. Backprop through  loss → unet → vae encoder → x_adv → delta
#   Accumulate and average the resulting gradients.
#
# Inputs:  x_adv, delta (requires_grad=True), c_o, outer_iter_index i
# Output:  avg_grad  (same shape as delta)
# ─────────────────────────────────────────────────────────────────────────────

FUNCTION getAvgGrad(x_adv, delta, c_o, i):

    grad_accum ← zeros_like(delta)

    FOR _ in 1 .. M:                               # M = 10 timestep samples

        # --- encode x_adv (stochastic reparameterization) ---
        z_mean, z_std ← vae.encode(x_adv).latent_dist.{mean, std}
        z_adv         ← (z_mean + z_std * randn_like(z_mean)) * scaling_factor

        # --- forward diffuse to a random timestep ---
        t      ← randint(0, T)
        eps_t  ← randn_like(z_adv)
        z_t    ← sqrt(alpha_bar[t]) * z_adv + sqrt(1 - alpha_bar[t]) * eps_t

        # --- build reference latent (MSE target) ---
        ref    ← randn_like(z_mean) * scheduler.init_noise_sigma
        ref    ← scheduler.scale_model_input(ref, t)

        # --- choose text conditioning ---
        IF i % k == 0:
            c ← embedding_pgd(unet, z_mean, z_t, t, c_o)  # adversarial embedding
        ELSE:
            c ← c_o                                         # original embedding

        # --- UNet forward (input is deterministic mean latent, not z_adv) ---
        noise_pred ← unet(z_mean, t, encoder_hidden_states=c)

        # --- loss and gradient ---
        loss      ← MSE(noise_pred, ref)
        grad_accum ← grad_accum + ∂loss/∂delta     # backprop through vae encoder

    RETURN grad_accum / M


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP: TDAE_PhotoGD
#
# Two-gradient momentum scheme:
#   grad1  — gradient computed at current delta  (used for a preliminary step)
#   grad2  — gradient recomputed after the preliminary step
#   final update uses  a * grad1 + (1-a) * grad2,  best a = 0.7
#
# Inputs:  x0, unet, vae, c_o, scheduler
# Output:  x_adv_final
# ─────────────────────────────────────────────────────────────────────────────

FUNCTION TDAE(x0, c):

    δv ← 0
    δp ← 0
    ximu ← x0

    FOR n in 1 .. N:

        e ← c + δp

        # ---- DPD: periodically update text perturbation ----
        IF n % S == 0:
            δp ← 0
            FOR m in 1 .. M_dpd:
                e_dpd   ← c + δp
                L_dpd   ← Loss(x0 + δv, e_dpd)         # or ximu
                g_p     ← ∂L_dpd / ∂δp
                δp      ← δp - η * sign(g_p)
                δp      ← project_Linf(δp, ϵp)
            END FOR
            e ← c + δp
        END IF

        # ---- FDM on image perturbation ----
        L1, g1 ← LossAndGrad(δv, e)                    # at current point
        s      ← g1 / ||g1||_2
        δv'    ← δv + h * s

        L2, g2 ← LossAndGrad(δv', e)                   # at perturbed point

        z      ← L2 - L1
        gFDM   ← -g1 + (λ / h) * sign(z) * (g2 - g1)

        δv     ← δv - α * sign(gFDM)
        δv     ← project_Linf(δv, ϵv)
        δv     ← clamp(x0 + δv, 0, 1) - x0

        ximu   ← x0 + δv

    RETURN x0 + δv
