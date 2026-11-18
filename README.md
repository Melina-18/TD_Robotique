# TD Robotique


## Partie 2 : LeRobot


Voici les vidéos du robot:  

https://github.com/user-attachments/assets/6c097975-c87c-44bb-adbf-c9d0f23e321f

https://github.com/user-attachments/assets/eb477b81-6e69-4ba7-bfd6-5f6a1882882e  

https://github.com/user-attachments/assets/9c53638a-34c6-43a2-958b-94a95acc1355  

https://github.com/user-attachments/assets/7171d93b-7e13-4994-9ac3-b50cdbf5f647  

Le retour a sa position initiale :  

https://github.com/user-attachments/assets/becbd2c6-1f79-417c-afc1-f7b2befdfa70 

## I - LQR

Le LQR (*Linear Quadratic Regulator*) est un contrôleur par retour d'état qui cherche à amener ou maintenir un système à l'équilibre, le plus efficacement possible. La loi de contrôle est un retour d'état :

$$
u(t) = -K\,x(t)
$$

On a une fonction de coût

$$
J = \int_{0}^{\infty} \left( x^{T} Q x + u^{T} R u \right) dt
$$

La solution optimale est un retour d'état $u = \bar{\bar{u}} - K(x - \bar{\bar{x}})$, avec $K = R^{-1} B^{T} P$, où $P$ est la solution de l'équation algébrique de Riccati :

$$
A^{T} P + P A - P B R^{-1} B^{T} P + Q = 0
$$

Pour utiliser le LQR on a besoin d'un modèle mathématique du robot. Nous n'avons pas construit les modèles du robot nous-mêmes : ils viennent de la bibliothèque `lerobot-kinematics` :

- **Modèle cinématique.** `get_robot('so100')`, dans `lerobot_Kinematics.py`.
- **Modèle MuJoCo.** `so_100.xml` et `scene.xml`.

### I.1 Modèle $\dot{x} = Ax + Bu$

L'état contient les 6 positions et les 6 vitesses articulaires. $M$ est la matrice d'inertie lue dans MuJoCo dans la posture initiale et $D$ l'amortissement des articulations :

$$
M\ddot{q} + D\dot{q} = u
\quad \Longrightarrow \quad
\dot{x} =
\begin{bmatrix} 0 & I \\ 0 & -M^{-1}D \end{bmatrix} x
+
\begin{bmatrix} 0 \\ M^{-1} \end{bmatrix} u
$$

```python
M_full = np.zeros((mjmodel.nv, mjmodel.nv))
mujoco.mj_fullM(mjmodel, M_full, mjdata.qM)
M_robot = M_full[np.ix_(qvel_indices, qvel_indices)]
M_inv = np.linalg.inv(M_robot)
D_robot = np.diag(mjmodel.dof_damping[qvel_indices])

A_mat = np.zeros((n_states, n_states))
A_mat[:dof, dof:] = np.eye(dof)
A_mat[dof:, dof:] = -M_inv @ D_robot

B_mat = np.zeros((n_states, dof))
B_mat[dof:, :] = M_inv

C_mat = np.eye(n_states)
D_mat = np.zeros((n_states, dof))

sys_lineare = statespace.StateSpaceSystem(A_mat, B_mat, C_mat, D_mat)
```

### I.2 Choix de $Q$ et $R$

- **$Q$ grand** : l'écart coûte cher. Le mouvement est plus rapide et plus précis, avec des commandes plus fortes ;
- **$R$ grand** : l'effort coûte cher. Le mouvement est plus lent et plus doux.

Ce qui compte est le rapport $Q/R$ : multiplier $Q$ et $R$ par le même nombre donne le même $K$.

```python
def guadagno_lqr(sys_lineare, facteur_q_pos, facteur_q_vel, r):
    cf = costfunction.QuadraticCostFunction.from_sys(sys_linear)
    cf.Q = np.diag(np.concatenate([facteur_q_pos * Q_POS,
                                   facteur_q_vel * Q_VEL]))
    cf.R = r * np.eye(dof)
    xbar_init = np.concatenate([init_qpos, np.zeros(dof)])
    ctl_lqr = lqr.synthesize_lqr_controller(sys_lineare, cf,
                                            xbar=xbar_init,
                                            ubar=np.zeros(dof))
    poli = np.linalg.eigvals(sys_lineare.A - sys_lineare.B @ ctl_lqr.K)
    return ctl_lqr, poli
```

Après, dans le code, on utilise cette fonction pour les différentes valeurs de $Q$ et $R$. On fait 4 tests différents pour voir les changements sur $Q_{POS}$, sur $Q_{VEL}$ et sur $R$ séparément.

<p align="center">
  <img src="https://github.com/user-attachments/assets/03433707-37e6-4f4f-b7dd-99ab2f72b5da" width="48%">
  <img src="https://github.com/user-attachments/assets/3b5a016d-12f0-4a3b-9dbb-a0e788d5ed36" width="48%">
</p>



### I.3 Évitement d'une zone

Pour l'évitement d'une zone, le code vérifie déjà que le trajet ne descend pas sous une hauteur minimale avant de bouger le robot : le même contrôle pourrait interdire un volume donné.


## II - DMP

Un DMP est un système dynamique stable qui « ressort » vers un but, auquel on ajoute un terme appris qui donne sa forme au mouvement. Il a été introduit par Ijspeert, Nakanishi et Schaal (2002, 2013) pour apprendre des gestes par démonstration.
Son intérêt : une seule démonstration suffit, le mouvement converge toujours vers le but, et on peut changer le but, la durée ou le point de départ sans réapprendre.

### II.1 Le système de transformation
Chaque dimension du mouvement (position y) suit un ressort-amortisseur, plus un terme de forçage f :
\tau \dot{v} = \alpha_z \left( \beta_z (g - y) - v \right) + f(s)
\tau \dot{y} = v
• g : le but (position finale) ; y0 : la position de départ ;
• τ : la durée du mouvement (le diviser accélère tout) ;
• αz (« alpha ») : l'amortissement, il règle la rapidité de réaction ;
• βz (« beta ») : la raideur du ressort vers le but.
Sans f, ce système rejoint simplement g. Avec β = α/4, il est amorti de façon critique : il arrive au but le plus vite possible sans le dépasser. Le facteur d'amortissement vaut ζ = ½ × √(α/β) : ζ > 1 est suramorti (lent), ζ < 1 sous-amorti (oscillations).

### II.2 Le système canonique (l'horloge)
Le temps est remplacé par une phase s qui décroît de 1 vers 0 :
\tau \dot{s} = -\alpha_s \, s
Toutes les dimensions partagent cette même phase, ce qui les garde synchronisées. αs règle la vitesse à laquelle s s'éteint.







