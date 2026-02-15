Oracles Module
==============

Bug detection oracles for automated game QA. Each oracle monitors an episode
in real time and produces structured findings (potential bugs) that are
aggregated into the episode report.

.. automodule:: src.oracles
   :no-members:

Oracle (Base Class)
-------------------

.. autoclass:: src.oracles.Oracle
   :members:
   :undoc-members:
   :show-inheritance:

Finding
-------

.. autoclass:: src.oracles.base.Finding
   :members:
   :undoc-members:
   :no-index:

CrashOracle
-----------

.. autoclass:: src.oracles.CrashOracle
   :members:
   :undoc-members:
   :show-inheritance:

StuckOracle
-----------

.. autoclass:: src.oracles.StuckOracle
   :members:
   :undoc-members:
   :show-inheritance:

ScoreAnomalyOracle
------------------

.. autoclass:: src.oracles.ScoreAnomalyOracle
   :members:
   :undoc-members:
   :show-inheritance:

VisualGlitchOracle
------------------

.. autoclass:: src.oracles.VisualGlitchOracle
   :members:
   :undoc-members:
   :show-inheritance:

PerformanceOracle
-----------------

.. autoclass:: src.oracles.PerformanceOracle
   :members:
   :undoc-members:
   :show-inheritance:

PhysicsViolationOracle
----------------------

.. autoclass:: src.oracles.PhysicsViolationOracle
   :members:
   :undoc-members:
   :show-inheritance:

BoundaryOracle
--------------

.. autoclass:: src.oracles.BoundaryOracle
   :members:
   :undoc-members:
   :show-inheritance:

StateTransitionOracle
---------------------

.. autoclass:: src.oracles.StateTransitionOracle
   :members:
   :undoc-members:
   :show-inheritance:

EpisodeLengthOracle
-------------------

.. autoclass:: src.oracles.EpisodeLengthOracle
   :members:
   :undoc-members:
   :show-inheritance:

TemporalAnomalyOracle
----------------------

.. autoclass:: src.oracles.TemporalAnomalyOracle
   :members:
   :undoc-members:
   :show-inheritance:

RewardConsistencyOracle
-----------------------

.. autoclass:: src.oracles.RewardConsistencyOracle
   :members:
   :undoc-members:
   :show-inheritance:

SoakOracle
----------

.. autoclass:: src.oracles.SoakOracle
   :members:
   :undoc-members:
   :show-inheritance:
