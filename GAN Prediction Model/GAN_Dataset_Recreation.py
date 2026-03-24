import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lambert_jonas(vi, a=1.0, p=3.0, v_bl=200.0):
    vi = np.asarray(vi, dtype=float)
    vr = np.zeros_like(vi, dtype=float)
    mask = vi > v_bl
    vr[mask] = a * (vi[mask] ** p - v_bl ** p) ** (1.0 / p)
    return vr

class Scaler:
    def __init__(self, vmax=1200.0):
        self.vmax = float(vmax)
    def transform(self, X):
        return X / self.vmax
    def inverse(self, Xs):
        return Xs * self.vmax

def make_generator(latent_dim=32, n_classes=5):
    z_in = Input(shape=(latent_dim,), name="z_in")
    vi_in = Input(shape=(1,), name="vi_in")            
    y_in = Input(shape=(1,), dtype="int32", name="y_in")
    # Embed class label
    y_emb = layers.Embedding(n_classes, 8)(y_in)
    y_emb = layers.Flatten()(y_emb)
    x = layers.Concatenate()([z_in, vi_in, y_emb])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    # Output a single v_r_scaled in [0,1] (sigmoid)
    vr_out = layers.Dense(1, activation="relu", name="v_r_out")(x)

    return Model([z_in, vi_in, y_in], vr_out, name="Generator_cond_vi")

def make_discriminator(n_classes=5):
    vi_in = Input(shape=(1,), name="vi_in")
    vr_in = Input(shape=(1,), name="vr_in")
    y_in = Input(shape=(1,), dtype="int32", name="y_in")
    y_emb = layers.Embedding(n_classes, 8)(y_in)
    y_emb = layers.Flatten()(y_emb)
    x = layers.Concatenate()([vi_in, vr_in, y_emb])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model([vi_in, vr_in, y_in], out, name="Discriminator_cond_vi")
  
def load_data_from_excel(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Threat Level 6')
    # Extract data
    vi = df['velocity'].values
    vr = df['residual velocity'].values
    # Create X array with vi and vr pairs
    X = np.column_stack([vi, vr]).astype(np.float32)
    # Single class (all zeros)
    Y = np.zeros(len(vi), dtype=np.int32)
    return X, Y

def train_and_generate_with_fit(classes_vbl=None,samples_per_class=800,vi_max=1200.0,latent_dim=32,batch_size=64,epochs=3000,seed=42,class_to_generate=1,save_excel=False):
    if classes_vbl is None:
        classes_vbl = [400, 500, 600, 700, 800]
    n_classes = len(classes_vbl)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # prepare dataset with uniform vi sampling
    print("Preparing synthetic dataset...")
    X_raw, Y, _ = make_ballistic_dataset(classes_vbl, samples_per_class, vi_max=vi_max)
    scaler = Scaler(vmax=vi_max * 1.05)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    vi_scaled_all = X_scaled[:, 0:1]
    vr_scaled_all = X_scaled[:, 1:2]
    # build models
    G = make_generator(latent_dim=latent_dim, n_classes=n_classes)
    D = make_discriminator(n_classes=n_classes)
    # compile
    d_lr = 1e-4
    g_lr = 2e-4
    d_opt = Adam(learning_rate=d_lr, beta_1=0.5)
    g_opt = Adam(learning_rate=g_lr, beta_1=0.5)
    D.compile(optimizer=d_opt, loss="binary_crossentropy")
    z_input = Input(shape=(latent_dim,), name="z_comb")
    vi_input = Input(shape=(1,), name="vi_comb")
    y_input = Input(shape=(1,), dtype="int32", name="y_comb")
    fake_vr = G([z_input, vi_input, y_input])
    D.trainable = False
    validity = D([vi_input, fake_vr, y_input])
    GAN = Model([z_input, vi_input, y_input], validity)
    GAN.compile(optimizer=g_opt, loss="binary_crossentropy")
    D.trainable = True
    n_samples = vi_scaled_all.shape[0]
    half_batch = batch_size // 2
    real_label = 0.9
    fake_label = 0.0

    print(f"Starting training: epochs={epochs}, samples={n_samples}, classes={classes_vbl}")

    # training loop
    for t in range(1, epochs + 1):
        #Train D on real samples
        idx = np.random.randint(0, n_samples, half_batch)
        vi_real = vi_scaled_all[idx]
        vr_real = vr_scaled_all[idx]
        y_real = Y[idx].reshape(-1, 1)
        labels_real = np.full((half_batch, 1), real_label, dtype=np.float32)
        # Train D on fake samples
        z = np.random.normal(0, 1, (half_batch, latent_dim)).astype(np.float32)
        y_fake = np.random.randint(0, n_classes, size=(half_batch, 1)).astype(np.int32)
        # sample vi uniformly for fake so generator learns mapping across range
        vi_fake = np.random.uniform(0.0, 1.0, size=(half_batch, 1)).astype(np.float32)
        vr_fake = G.predict([z, vi_fake, y_fake], verbose=0)
        labels_fake = np.full((half_batch, 1), fake_label, dtype=np.float32)
        d_loss_real = D.train_on_batch([vi_real, vr_real, y_real], labels_real)
        d_loss_fake = D.train_on_batch([vi_fake, vr_fake, y_fake], labels_fake)
        #Train G to fool D
        z2 = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        vi_g = np.random.uniform(0.0, 1.0, size=(batch_size, 1)).astype(np.float32)
        y_g = np.random.randint(0, n_classes, size=(batch_size, 1)).astype(np.int32)
        trick = np.full((batch_size, 1), real_label, dtype=np.float32)
        D.trainable = False
        g_loss = GAN.train_on_batch([z2, vi_g, y_g], trick)
        D.trainable = True

        if t % 250 == 0 or t == 1:
            print(f"Iter {t}: D_real={d_loss_real:.4f}, D_fake={d_loss_fake:.4f}, G={g_loss:.4f}")

    def get_training_data_for_class(class_id):
        mask = (Y == class_id)
        vi_train = X_raw[mask, 0]
        vr_train = X_raw[mask, 1]
        # sort by vi for clean comparison
        idx = np.argsort(vi_train)
        return vi_train[idx], vr_train[idx]

    vi_train, vr_train = get_training_data_for_class(class_to_generate)

    def generate_for_class(class_id, n_points=3000):
        vi_sweep = np.linspace(0.0, vi_max, n_points).astype(np.float32).reshape(-1,1)
        vi_sweep_scaled = scaler.transform(vi_sweep)
        z = np.random.normal(0, 1, (n_points, latent_dim)).astype(np.float32)
        labels = np.full((n_points, 1), class_id, dtype=np.int32)
        vr_pred_scaled = G.predict([z, vi_sweep_scaled, labels], verbose=0)
        vr_pred = scaler.inverse(vr_pred_scaled)
        vi_orig = vi_sweep.reshape(-1)
        vr_pred = np.clip(vr_pred, 0.0, None).reshape(-1)
        vr_pred = np.minimum(vr_pred, vi_orig)
        return vi_orig, vr_pred
      
    def generate_for_given_vi(class_id, vi_values):
        vi_scaled = scaler.transform(vi_values.reshape(-1,1))
        z = np.random.normal(0, 1, (len(vi_values), latent_dim)).astype(np.float32)
        labels = np.full((len(vi_values), 1), class_id, dtype=np.int32)
        vr_scaled = G.predict([z, vi_scaled, labels], verbose=0)
        vr_pred = scaler.inverse(vr_scaled).reshape(-1)
        vr_pred = np.clip(vr_pred, 0.0, None)
        vr_pred = np.minimum(vr_pred, vi_values)
        return vr_pred

    vr_gan_on_train_vi = generate_for_given_vi(class_to_generate, vi_train)

    class_to_generate = 1
    print(f"\nGenerating predictions (sweep) for class {class_to_generate} (v_bl={classes_vbl[class_to_generate]})...")
    vi_gen, vr_gen = generate_for_class(class_to_generate, n_points=4000)

    mask = vr_gen > 0.0
    vi_fit = vi_gen[mask]
    vr_fit = vr_gen[mask]
    p0 = [1.0, 3.0, float(classes_vbl[class_to_generate])]
    lb = [1e-4, 1.0, 0.0]
    ub = [20.0, 10.0, vi_max * 1.05]

    a_fit = p_fit = vbl_fit = np.nan
    try:
        popt, pcov = curve_fit(lambert_jonas, vi_fit, vr_fit, p0=p0, bounds=(lb, ub), maxfev=40000)
        a_fit, p_fit, vbl_fit = popt
        print(f"\nFitted Lambert–Jonas from GAN-generated points: a={a_fit:.4f}, p={p_fit:.4f}, v_bl={vbl_fit:.2f}")
    except Exception as e:
        print("Curve fit failed:", e)
    # compute fitted curve values
    if not np.isnan(a_fit):
        vr_gen_fit = lambert_jonas(vi_gen, a=a_fit, p=p_fit, v_bl=vbl_fit)
    else:
        vr_gen_fit = np.full_like(vr_gen, np.nan)
    vbl_true = classes_vbl[class_to_generate]
    vbl_pred = vbl_fit
    print("\nComparison: Training vs GAN-generated (sampled)")
    print("{:>10s} {:>18s} {:>18s}".format("v_i", "v_r_true", "v_r_GAN"))
    print("-" * 50)

    step = max(1, len(vi_train)//50)
    for i in range(0, len(vi_train),step):
      print(f"{vi_train[i]:10.2f} {vr_train[i]:18.2f} {vr_gan_on_train_vi[i]:18.2f}")

    plt.figure(figsize=(11, 7))
    plt.scatter(vi_train, vr_gan_on_train_vi,s=28, color="red", alpha=0.55,label="GAN prediction (same $v_i$)")
    vi_line = np.linspace(0, vi_max, 800)
    vr_true = lambert_jonas(vi_line, v_bl=vbl_true)
    plt.plot(vi_line, vr_true,"k-", linewidth=2.2,label=f"True Lambert–Jonas (V_bl={vbl_true:.0f} m/s)")
    if not np.isnan(vbl_pred):
        plt.plot(vi_gen, vr_gen_fit,color="darkred", linewidth=2.5,label=f"Fitted Lambert–Jonas (V_bl={vbl_pred:.1f} m/s)")
    plt.axvline(vbl_true,color="green", linestyle="-", linewidth=2.5,label=f"True V_bl = {vbl_true:.0f} m/s")
    if not np.isnan(vbl_pred):
        plt.axvline(vbl_pred,color="red", linestyle="--", linewidth=2.5,label=f"Predicted V_bl = {vbl_pred:.1f} m/s")
    plt.xlabel("Impact Velocity $v_i$ (m/s)", fontsize=12)
    plt.ylabel("Residual Velocity $v_r$ (m/s)", fontsize=12)
    plt.title(f"GAN | Class {class_to_generate}\n" f"True V_bl={vbl_true:.0f} m/s, Predicted V_bl={vbl_pred:.1f} m/s",fontsize=13)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"G": G,"D": D,"vi_gen": vi_gen,"vr_gen": vr_gen,"classes_vbl": classes_vbl,"fit_params": (a_fit, p_fit, vbl_fit)}

if __name__ == "__main__":
    results = train_and_generate_with_fit(classes_vbl=[400, 500, 600, 700, 800],samples_per_class=800,vi_max=1200.0,latent_dim=32,batch_size=64,epochs=5000,seed=2025)
    print("\nDone. Fitted params:", results["fit_params"])
