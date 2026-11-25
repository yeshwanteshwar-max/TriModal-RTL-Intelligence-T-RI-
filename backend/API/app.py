from fastapi import FastAPI

from .routes import (
    analysis_routes,
    design_routes,
    multimodal_routes,
    optimization_routes,
    report_routes,
    simulation_routes,
)

app = FastAPI(title="RTL AI Studio", version="1.0.0")
app.include_router(design_routes.router)
app.include_router(multimodal_routes.router)
app.include_router(analysis_routes.router)
app.include_router(simulation_routes.router)
app.include_router(optimization_routes.router)
app.include_router(report_routes.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
