import numpy as np
from onnx.reference.ops.op_scan import Scan as _Scan


class Scan(_Scan):

    def need_context(self) -> bool:
        """Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Loop).
        The default answer is `False`.
        """
        return True

    def _run(
        self,
        *args,
        context=None,
        body=None,
        num_scan_inputs=None,
        scan_input_axes=None,
        scan_input_directions=None,
        scan_output_axes=None,
        scan_output_directions=None,
        attributes=None,
    ):
        (
            num_loop_state_vars,
            _num_scan_outputs,
            _output_directions,
            _max_dir_out,
            _output_axes,
            _max_axe_out,
            state_names_in,
            state_names_out,
            scan_names_in,
            scan_names_out,
            scan_values,
            states,
        ) = self._common_run_shape(*args)

        max_iter = args[num_loop_state_vars].shape[self.input_axes_[0]]
        results = [[] for _ in scan_names_out]  # type: ignore

        for it in range(max_iter):
            inputs = context.copy()
            inputs.update(dict(zip(state_names_in, states)))
            inputs.update({name: value[it] for name, value in zip(scan_names_in, scan_values)})

            try:
                outputs_list = self._run_body(inputs)  # type: ignore
            except TypeError as e:
                raise TypeError(
                    f"Unable to call 'run' for type '{type(self.body)}'."  # type: ignore
                ) from e

            outputs = dict(zip(self.output_names, outputs_list))
            states = [outputs[name] for name in state_names_out]
            for i, name in enumerate(scan_names_out):
                results[i].append(np.expand_dims(outputs[name], axis=0))

        for res in results:
            conc = np.vstack(res)
            states.append(conc)
        return self._check_and_fix_outputs(tuple(states))
