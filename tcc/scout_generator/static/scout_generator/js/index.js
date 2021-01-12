$(document).ready(function(){
    let $radio = $('input[name="predict-option"]');
    $radio.click(function(){        
        let option = $('input[name="predict-option"]:checked').val(); 
        if (option == 'event-identification') {
            $('.scout-form').hide();
            $('.predict-form').show();
        } else {
            $('.predict-form').hide();
            $('.scout-form').show();
        }
    });
});